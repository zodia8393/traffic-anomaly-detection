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
        if any(k in t for k in ("가장심각", "사망", "최다사상", "인명피해가큰", "대형사고", "가장큰사고")):
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
        """웹/API용 구조화 응답: {answer, stats, cases}. 의도별 라우팅."""
        intent = self._intent(question)
        if intent != "semantic" and self._db is not None:
            out = self._structured_answer(intent, question, top_k)
            if out is not None:
                return out
        return self._semantic_answer(question, top_k)

    # ── 구조화 질의 (SQL) ────────────────────────────────────────────
    def _structured_answer(self, intent: str, q: str, top_k: int) -> dict | None:
        line = self._line_filter(q)
        where = "WHERE acc_dt IS NOT NULL"
        params: list = []
        scope = ""
        if line:
            where += " AND line_name = ?"
            params.append(line)
            scope = f"{line} "
        try:
            if intent == "count":
                where_c = "WHERE 1=1" + (" AND line_name = ?" if line else "")
                n = self._db.execute(f"SELECT COUNT(*) FROM chunks {where_c}", params).fetchone()[0]
                by = self._db.execute(
                    f"SELECT document_type, COUNT(*) c FROM chunks {where_c} "
                    f"GROUP BY document_type ORDER BY c DESC", params).fetchall()
                bd = ", ".join(f"{t} {c}건" for t, c in by if t)
                ans = f"{scope}사고는 총 {n}건 기록돼 있습니다." + (f" (유형별: {bd})" if bd else "")
                cases = self._recent_cases(line, 3)
                return {"answer": ans, "stats": {"n": n}, "cases": cases}

            order = {"recent": "acc_dt DESC", "longest": "closure_min DESC",
                     "severe": "casualties DESC, closure_min DESC"}[intent]
            extra = "" if intent == "recent" else (" AND closure_min IS NOT NULL" if intent == "longest"
                                                   else " AND casualties IS NOT NULL")
            rows = self._db.execute(
                f"SELECT acc_dt, line_name, direction, document_type, vehicles, cause, "
                f"closure_min, casualties, chunk_text FROM chunks {where}{extra} "
                f"ORDER BY {order} LIMIT {top_k}", params).fetchall()
            if not rows:
                return None
            top = rows[0]
            dt, ln, dr, ty, veh, cau, clo, cas, txt = top
            head = {"recent": f"가장 최근 {scope}사고는 {dt}에 발생했습니다.",
                    "longest": f"차단시간이 가장 길었던 {scope}사고입니다.",
                    "severe": f"인명피해가 가장 컸던 {scope}사고입니다."}[intent]
            detail = f"{ln} {dr}방향 {ty}, 피해차량 {veh or '미상'}"
            if cau:
                detail += f", 원인 {cau}"
            if clo is not None:
                detail += f", 차단/처리 {clo}분"
            if cas:
                detail += f", 인명피해 {cas}명"
            ans = f"{head}\n{detail}."
            cases = [{"text": r[8], "score": 1.0 - i * 0.05} for i, r in enumerate(rows)]
            stat = ({"max": int(top[6]), "n": len(rows)} if intent == "longest"
                    else {"인명피해": int(top[7]), "n": len(rows)} if intent == "severe" else None)
            return {"answer": ans, "stats": stat, "cases": cases}
        except Exception:  # noqa: BLE001 — 구조화 실패 시 의미검색으로 폴백
            return None

    def _recent_cases(self, line, k):
        w = "WHERE acc_dt IS NOT NULL" + (" AND line_name = ?" if line else "")
        p = [line] if line else []
        rows = self._db.execute(f"SELECT chunk_text FROM chunks {w} ORDER BY acc_dt DESC LIMIT {k}", p).fetchall()
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
        lines = [f"유사 사고 {len(hits)}건을 찾았습니다."]
        if closures:
            arr = np.array(closures)
            lines.append(
                f"실제 차단/처리시간: 평균 {arr.mean():.0f}분, 중앙값 {int(np.median(arr))}분 "
                f"(범위 {arr.min()}~{arr.max()}분, n={len(closures)}).")
        lines.append("\n가장 유사한 사례:")
        for i, h in enumerate(hits[:3], 1):
            lines.append(f"  {i}. {h['excerpt'][:140]}")
        return "\n".join(lines)

    def _llm_answer(self, question: str, hits: list[dict], closures: list[int]) -> str:
        if self._client is None:
            from layer3_mllm.mllm_client import MLLMClient
            self._client = MLLMClient(backend="transformers")
        ctx = "\n".join(f"- {h['excerpt'][:160]}" for h in hits[:5])
        stat = (f"유사사고 차단시간 평균 {np.mean(closures):.0f}분" if closures else "")
        prompt = (f"당신은 고속도로 사고 처리 전문가입니다. 아래 유사 실사고 기록만 근거로 "
                  f"질문에 간결히(3문장 이내) 한국어로 답하시오. 기록에 없는 내용은 추측 금지.\n\n"
                  f"질문: {question}\n\n유사 실사고:\n{ctx}\n{stat}\n\n답변:")
        res = self._client.chat([{"role": "user", "content": prompt}], max_tokens=256)
        c = res.get("content")
        return c if isinstance(c, str) else str(c)


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
