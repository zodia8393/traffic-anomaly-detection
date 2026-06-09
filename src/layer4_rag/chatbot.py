"""사고 차단시간 RAG 챗봇 — 벡터검색 + 차단시간 통계 기반 답변.

협업자 차단시간 DB로 만든 벡터 RAG(rag_knowledge.duckdb)에 질문하면, 유사 실사고를
의미검색으로 회수하고 실제 차단시간 통계로 근거 있는 답을 준다. CPU에서 즉시 동작
(LLM 생성 없이 검색+집계). --llm 플래그 시 MLLM이 대화형 문장 생성.

실행:
  python chatbot.py                 # 대화형 REPL
  python chatbot.py --demo          # 샘플 질문 데모
  python chatbot.py -q "경부선 화물차 전복 차단시간"
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from layer4_rag.rag_retriever import get_retriever

_CLOSURE_RE = re.compile(r"차단/?처리시간 약 (\d+)\s*분")


class RagChatbot:
    def __init__(self, use_llm: bool = False) -> None:
        self.r = get_retriever()
        self.use_llm = use_llm
        self._client = None
        # 구조화 질의(최근/최대/건수)용 read-only 연결
        self._db = None
        try:
            import duckdb
            from config_new import RAG_DB_PATH
            if Path(RAG_DB_PATH).exists():
                self._db = duckdb.connect(RAG_DB_PATH, read_only=True)
        except Exception:  # noqa: BLE001
            self._db = None

    # ── 의도 분류 (구조화 vs 의미) ───────────────────────────────────
    @staticmethod
    def _intent(q: str) -> str:
        t = q.replace(" ", "")
        if any(k in t for k in ("최근", "최신", "마지막", "언제", "요즘")):
            return "recent"
        if any(k in t for k in ("가장긴", "제일긴", "가장오래", "제일오래", "최대차단", "가장길", "오래걸린", "longest")):
            return "longest"
        if any(k in t for k in ("몇건", "몇개", "건수", "개수", "총몇", "얼마나많", "총건")):
            return "count"
        if (("인명피해" in t or "사상자" in t) and any(k in t for k in ("큰", "많", "최대", "가장"))) \
                or any(k in t for k in ("가장심각", "사망자가많", "최다사상", "대형사고", "가장큰사고")):
            return "severe"
        return "semantic"

    def _line_filter(self, q: str):
        """질문에서 노선명 추출(있으면 SQL 필터)."""
        if self._db is None:
            return None
        try:
            lines = [r[0] for r in self._db.execute(
                "SELECT DISTINCT line_name FROM chunks WHERE line_name IS NOT NULL").fetchall()]
        except Exception:  # noqa: BLE001
            return None
        qn = q.replace(" ", "")
        for ln in sorted(lines, key=len, reverse=True):
            if ln and ln.replace(" ", "") in qn:
                return ln
        return None

    @property
    def available(self) -> bool:
        return self.r.available

    def ask(self, question: str, top_k: int = 5) -> str:
        hits = self.r.search(question, top_k=top_k)
        if not hits:
            return "관련 사고 기록을 찾지 못했습니다. (질문을 노선·사고유형·차량 중심으로 바꿔보세요)"
        # 차단시간 추출(집계)
        closures = []
        for h in hits:
            m = _CLOSURE_RE.search(h.get("excerpt", "") or h.get("chunk_text", ""))
            if m:
                closures.append(int(m.group(1)))
        if self.use_llm:
            return self._llm_answer(question, hits, closures)
        return self._grounded_answer(hits, closures)

    def ask_structured(self, question: str, top_k: int = 5) -> dict:
        """하이브리드 지능: 구조화 SQL 사실 + 벡터 RAG 사례 + 통계 → (LLM)종합 답변.

        - 정확한 사실(최근/건수/최대)은 SQL로, 의미 사례는 벡터로, 차단시간은 통계로.
        - use_llm이면 이 근거들을 LLM이 자연어로 종합(숫자 그대로 인용, 환각 억제).
        - 아니면 구조화 답변(있으면) 또는 통계 요약(빠름).
        """
        intent = self._intent(question)
        struct = (self._structured_answer(intent, question, top_k)
                  if intent != "semantic" and self._db is not None else None)
        # 의미 사례 + 차단시간 통계 (항상 — LLM 근거/폴백용)
        hits = self.r.search(question, top_k=top_k)
        closures = [int(m.group(1)) for h in hits
                    if (m := _CLOSURE_RE.search(h.get("excerpt", "") or h.get("chunk_text", "")))]
        sem_stats = None
        if closures:
            a = np.array(closures)
            sem_stats = {"n": len(closures), "avg": round(float(a.mean())),
                         "median": int(np.median(a)), "min": int(a.min()), "max": int(a.max())}

        if struct:
            fact, stats, cases = struct["answer"], struct.get("stats"), struct["cases"]
        elif hits:
            fact, stats = None, sem_stats
            cases = [{"text": h.get("excerpt", ""), "score": h.get("score", 0)} for h in hits]
        else:
            return {"answer": "관련 사고 기록을 못 찾았어요. 노선·사고유형·차량 중심으로 물어봐 주세요.",
                    "stats": None, "cases": []}

        if self.use_llm:
            answer = self._smart_llm_answer(question, fact, cases, stats or sem_stats)
        else:
            answer = fact if fact else self._grounded_answer(hits, closures)
        return {"answer": answer, "stats": stats, "cases": cases}

    def _smart_llm_answer(self, question: str, fact: str | None,
                          cases: list[dict], stats: dict | None) -> str:
        """구조화 사실 + 사례 + 통계를 LLM이 근거기반으로 종합."""
        if self._client is None:
            from layer3_mllm.mllm_client import MLLMClient
            self._client = MLLMClient(backend="transformers")
        parts = []
        if fact:
            parts.append(f"[확인된 사실]\n{fact}")
        if cases:
            parts.append("[관련 실사고 기록]\n" + "\n".join(f"- {c['text'][:150]}" for c in cases[:5]))
        if stats and stats.get("avg") is not None:
            parts.append(f"[차단시간 통계] 평균 {stats['avg']}분, 중앙값 {stats.get('median')}분, "
                         f"범위 {stats.get('min')}~{stats.get('max')}분 (n={stats.get('n')})")
        ctx = "\n\n".join(parts) or "(근거 없음)"
        prompt = ("당신은 고속도로 사고·차단시간 분석 전문가입니다. 아래 근거만 사용해 질문에 "
                  "한국어로 간결하고 자연스럽게(2~4문장) 답하시오. 숫자·날짜·노선명은 근거 그대로 "
                  "인용하고, 근거에 없는 내용은 추측하지 마시오. 마지막에 핵심 수치를 한 번 더 짚어주면 좋습니다.\n\n"
                  f"{ctx}\n\n질문: {question}\n답변:")
        res = self._client.chat([{"role": "user", "content": prompt}], max_tokens=320, parse_json=False)
        c = res.get("content")
        return (c if isinstance(c, str) else str(c)).strip() or (fact or "답변 생성에 실패했습니다.")

    # ── 복합 필터 빌더 (노선+유형+차량+원인) ─────────────────────────
    def _build_where(self, q: str):
        conds, params, scope = [], [], []
        line = self._line_filter(q)
        if line:
            conds.append("line_name = ?"); params.append(line); scope.append(line)
        t = q.replace(" ", "")
        for atype in ("전면차단", "부분차단", "갓길"):
            if atype in t:
                conds.append("document_type LIKE ?"); params.append(f"%{atype}%"); scope.append(atype); break
        for veh in ("화물차", "트레일러", "버스", "승용차", "이륜", "오토바이"):
            if veh in t:
                conds.append("vehicles LIKE ?"); params.append(f"%{veh}%"); scope.append(veh); break
        for cause in ("졸음", "과속", "주시태만", "안전거리", "역주행", "빗길", "결빙", "화재"):
            if cause in t:
                col = "chunk_text" if cause in ("화재",) else "cause"
                conds.append(f"{col} LIKE ?"); params.append(f"%{cause}%"); scope.append(cause); break
        where = " AND ".join(conds) if conds else "1=1"
        return where, params, " ".join(scope)

    # ── 구조화 질의 (SQL) ────────────────────────────────────────────
    def _structured_answer(self, intent: str, q: str, top_k: int) -> dict | None:
        flt, params, scope = self._build_where(q)
        sc = (scope + " ") if scope else ""
        try:
            if intent == "count":
                n = self._db.execute(f"SELECT COUNT(*) FROM chunks WHERE {flt}", params).fetchone()[0]
                by = self._db.execute(
                    f"SELECT document_type, COUNT(*) c FROM chunks WHERE {flt} "
                    f"GROUP BY document_type ORDER BY c DESC", params).fetchall()
                bd = ", ".join(f"{t} {c}건" for t, c in by if t)
                if n == 0:
                    return {"answer": f"'{scope}' 조건의 사고 기록은 없습니다.", "stats": {"n": 0}, "cases": []}
                ans = f"{sc}사고는 총 {n}건 기록돼 있습니다." + (f" (유형별: {bd})" if bd else "")
                cases = self._sql_cases(f"{flt} AND acc_dt IS NOT NULL", params, "acc_dt DESC", 3)
                return {"answer": ans, "stats": {"n": n}, "cases": cases}

            order = {"recent": "acc_dt DESC", "longest": "closure_min DESC",
                     "severe": "casualties DESC, closure_min DESC"}[intent]
            cond = flt + " AND acc_dt IS NOT NULL"
            if intent == "longest":
                cond += " AND closure_min IS NOT NULL"
            elif intent == "severe":
                cond += " AND casualties > 0"
            rows = self._db.execute(
                f"SELECT acc_dt, line_name, direction, document_type, vehicles, cause, "
                f"closure_min, casualties, chunk_text FROM chunks WHERE {cond} "
                f"ORDER BY {order} LIMIT {top_k}", params).fetchall()
            if not rows:
                return None
            dt, ln, dr, ty, veh, cau, clo, cas, _ = rows[0]
            head = {"recent": f"가장 최근 {sc}사고는 {dt}에 일어났습니다.",
                    "longest": f"{sc}사고 중 차단시간이 가장 길었던 건은 {clo}분입니다.",
                    "severe": f"{sc}사고 중 인명피해가 가장 컸던 건은 {cas}명입니다."}[intent]
            detail = f"{ln} {dr}방향 {ty}, 피해차량 {veh or '미상'}"
            if cau:
                detail += f", 원인은 {cau}"
            if clo is not None and intent != "longest":
                detail += f", 차단/처리 {clo}분"
            if cas and intent != "severe":
                detail += f", 인명피해 {cas}명"
            ans = f"{head} ({detail})"
            cases = [{"text": r[8], "score": 1.0 - i * 0.05} for i, r in enumerate(rows)]
            stat = ({"최대분": int(rows[0][6]), "n": len(rows)} if intent == "longest"
                    else {"인명피해": int(rows[0][7]), "n": len(rows)} if intent == "severe" else None)
            return {"answer": ans, "stats": stat, "cases": cases}
        except Exception:  # noqa: BLE001 — 구조화 실패 시 의미검색으로 폴백
            return None

    def _sql_cases(self, cond, params, order, k):
        rows = self._db.execute(
            f"SELECT chunk_text FROM chunks WHERE {cond} ORDER BY {order} LIMIT {k}", params).fetchall()
        return [{"text": r[0], "score": 1.0} for r in rows]

    # ── 의미 질의 (벡터 RAG) ─────────────────────────────────────────
    def _semantic_answer(self, question: str, top_k: int) -> dict:
        hits = self.r.search(question, top_k=top_k)
        closures = []
        for h in hits:
            m = _CLOSURE_RE.search(h.get("excerpt", "") or h.get("chunk_text", ""))
            if m:
                closures.append(int(m.group(1)))
        stats = None
        if closures:
            arr = np.array(closures)
            stats = {"n": len(closures), "avg": round(float(arr.mean())),
                     "median": int(np.median(arr)), "min": int(arr.min()), "max": int(arr.max())}
        answer = (self._llm_answer(question, hits, closures) if (self.use_llm and hits)
                  else self._grounded_answer(hits, closures) if hits
                  else "관련 사고 기록을 찾지 못했습니다. 노선·사고유형·차량 중심으로 질문해 보세요.")
        cases = [{"text": h.get("excerpt", ""), "score": h.get("score", 0)} for h in hits]
        return {"answer": answer, "stats": stats, "cases": cases}

    def _grounded_answer(self, hits: list[dict], closures: list[int]) -> str:
        # 사례는 아래 카드로 표시되므로 답변은 자연스러운 요약만 (중복 제거)
        if closures:
            arr = np.array(closures)
            spread = f"{arr.min()}~{arr.max()}분" if arr.min() != arr.max() else f"{arr.min()}분"
            return (f"비슷한 사고 {len(hits)}건을 찾았어요. 이런 유형은 보통 차단·처리에 "
                    f"평균 {arr.mean():.0f}분(중앙값 {int(np.median(arr))}분, {spread}) 걸렸습니다. "
                    f"아래는 가장 유사한 사례들입니다.")
        return f"비슷한 사고 {len(hits)}건을 찾았어요. 아래 사례를 참고하세요."

    def _llm_answer(self, question: str, hits: list[dict], closures: list[int]) -> str:
        if self._client is None:
            from layer3_mllm.mllm_client import MLLMClient
            self._client = MLLMClient(backend="transformers")
        ctx = "\n".join(f"- {h['excerpt'][:160]}" for h in hits[:5])
        stat = (f"유사사고 차단시간 평균 {np.mean(closures):.0f}분" if closures else "")
        prompt = (f"당신은 고속도로 사고 처리 전문가입니다. 아래 유사 실사고 기록만 근거로 "
                  f"질문에 간결히(3문장 이내) 한국어로 답하시오. 기록에 없는 내용은 추측 금지.\n\n"
                  f"질문: {question}\n\n유사 실사고:\n{ctx}\n{stat}\n\n답변:")
        res = self._client.chat([{"role": "user", "content": prompt}], max_tokens=256,
                                parse_json=False)  # 챗봇은 평문 답변 (JSON 파싱·경고 불필요)
        c = res.get("content")
        return (c if isinstance(c, str) else str(c)).strip()


def _print_answer(q: str, a: str) -> None:
    print(f"\n\033[1m질문:\033[0m {q}")
    print(f"\033[36m답변:\033[0m {a}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--query")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--llm", action="store_true", help="MLLM 문장생성(느림, CPU)")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    bot = RagChatbot(use_llm=args.llm)
    if not bot.available:
        print("⚠️ RAG 지식베이스(rag_knowledge.duckdb) 미연결. RAG_DB_PATH 확인.")
        return
    print(f"🤖 사고 차단시간 RAG 챗봇 {'(LLM)' if args.llm else '(검색+통계)'} — 준비됨\n")

    if args.query:
        _print_answer(args.query, bot.ask(args.query, args.top_k))
        return
    if args.demo:
        for q in ["경부선 화물차 전복 적재물 유출 사고 차단시간",
                  "터널 화재 사고 처리시간",
                  "졸음운전 추돌 사고 사례와 차단시간",
                  "버스 다중추돌 인명피해 사고"]:
            _print_answer(q, bot.ask(q, args.top_k))
        return
    # 대화형 REPL
    print("질문을 입력하세요 (종료: quit/exit/q):")
    while True:
        try:
            q = input("\n\033[1m> \033[0m").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료."); break
        if q.lower() in ("quit", "exit", "q", ""):
            print("종료."); break
        print(f"\033[36m{bot.ask(q, args.top_k)}\033[0m")


if __name__ == "__main__":
    main()
