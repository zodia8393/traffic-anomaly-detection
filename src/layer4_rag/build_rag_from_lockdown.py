"""구조화 차단시간 DB(EX_LOCKDOWN_RAW) → 벡터 RAG DB 변환.

협업자의 차단시간 예측 DB(exkor_lockdown_*.db, RAG 청크 아님)를, 우리 벡터 RAG
리트리버(rag_retriever.py)가 바로 쓰는 스키마(documents/chunks/chunk_embeddings)로
변환한다. 각 실사고 행 → 자연어 청크(노선·유형·차량·원인·인명·정체 + 실제 차단시간) →
SBert(all-MiniLM-L6-v2) 임베딩 적재.

핵심: 사고접보 시각~사고처리 완료시간으로 **실제 차단시간(분)을 계산**해 청크에 포함 →
우리 CCTV가 사고 감지 시 유사 실사고 + 실제 차단시간을 벡터검색으로 회수.

실행:
  python build_rag_from_lockdown.py \
    --src ../../data/exkor_lockdown_20260526.db \
    --out ../../data/rag_knowledge.duckdb
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

import duckdb
import numpy as np

# 한국어 특화 임베더 — 영어 MiniLM은 한국어 분리력 부족(관련/무관 격차 0.12),
# ko-sroberta는 0.66으로 압도적(실측). 768d.
SBERT_MODEL = "jhgan/ko-sroberta-multitask"
SRC_TABLE = "EX_LOCKDOWN_RAW"


def _hhmm_to_min(s) -> int | None:
    if not s or not isinstance(s, str) or ":" not in s:
        return None
    try:
        h, m = s.split(":")[:2]
        return int(h) * 60 + int(m)
    except Exception:  # noqa: BLE001
        return None


def _closure_minutes(report_t, done_t) -> int | None:
    a, b = _hhmm_to_min(report_t), _hhmm_to_min(done_t)
    if a is None or b is None:
        return None
    d = b - a
    if d < 0:
        d += 24 * 60  # 자정 넘김
    return d if 0 < d < 24 * 60 else None


def _g(row: dict, key: str):
    v = row.get(key)
    return v if v not in (None, "", "nan") else None


def render_text(row: dict) -> tuple[str, str, list[str], int | None]:
    """사고 행 → (청크텍스트, document_type, keywords, closure_min)."""
    line = _g(row, "노선명") or "고속도로"
    direction = _g(row, "방향") or ""
    mp = _g(row, "이정")
    date = _g(row, "사고접보 시각(날짜)") or ""
    tm = _g(row, "사고접보 시각(시간)") or ""
    dow = _g(row, "사고접보 시각(요일)") or ""
    atype = _g(row, "사고 유형") or "사고"
    vehicles = [v for k in ("피해차량_1", "피해차량_2", "피해차량_3", "피해차량_4")
                if (v := _g(row, k))]
    cause = _g(row, "사고원인")
    total = _g(row, "인명피해(총수)")
    death = _g(row, "사망자수")
    serious = _g(row, "중상자수")
    minor = _g(row, "경상자수")
    fire = _g(row, "화재 여부")
    rollover = _g(row, "차량 전복/전도 여부")
    cargo = _g(row, "적재물 유출 여부")
    congest = _g(row, "정체길이")
    closure = _closure_minutes(tm, _g(row, "사고처리 완료시간"))

    parts = [f"{line} {direction}방향" + (f" {mp}km" if mp is not None else "") + f", {date} {tm} {dow}.".strip()]
    parts.append(f"사고유형: {atype}.")
    if vehicles:
        parts.append(f"피해차량: {', '.join(map(str, vehicles))}.")
    if cause:
        parts.append(f"사고원인: {cause}.")
    if total:
        cas = f"인명피해 {int(float(total))}명"
        sub = [f"{lbl}{int(float(v))}" for lbl, v in
               (("사망", death), ("중상", serious), ("경상", minor)) if v]
        if sub:
            cas += "(" + " ".join(sub) + ")"
        parts.append(cas + ".")
    flags = [f"화재 {fire}" if fire and "미" not in str(fire) else None,
             f"전복/전도 {rollover}" if rollover and "미" not in str(rollover) else None,
             f"적재물유출 {cargo}" if cargo and "미" not in str(cargo) else None]
    flags = [f for f in flags if f]
    if flags:
        parts.append(", ".join(flags) + ".")
    if congest is not None:
        parts.append(f"정체길이 {congest}km.")
    if closure is not None:
        parts.append(f"실제 차단/처리시간 약 {closure}분.")

    keywords = [k for k in (line, atype, cause, *(vehicles[:2])) if k]
    return " ".join(parts), atype, keywords, closure


def init_rag_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""CREATE TABLE IF NOT EXISTS documents(
        document_id VARCHAR PRIMARY KEY, file_name VARCHAR, domain VARCHAR,
        document_type VARCHAR, source VARCHAR)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS chunks(
        chunk_id VARCHAR PRIMARY KEY, document_id VARCHAR, chunk_text VARCHAR,
        keywords_json VARCHAR, domain VARCHAR, document_type VARCHAR,
        closure_min INTEGER)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS chunk_embeddings(
        embedding_id VARCHAR PRIMARY KEY, chunk_id VARCHAR, embedding_model VARCHAR,
        vector_dim INTEGER, embedding_json VARCHAR, is_current BOOLEAN)""")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../../data/exkor_lockdown_20260526.db")
    ap.add_argument("--out", default="../../data/rag_knowledge.duckdb")
    ap.add_argument("--table", default=SRC_TABLE)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    src = duckdb.connect(args.src, read_only=True)
    cols = [r[0] for r in src.execute(f"DESCRIBE {args.table}").fetchall()]
    rows = src.execute(f"SELECT * FROM {args.table}").fetchall()
    src.close()
    print(f"[load] {args.table}: {len(rows)}행, {len(cols)}컬럼", flush=True)

    # 텍스트 렌더
    chunks = []
    for r in rows:
        row = dict(zip(cols, r))
        text, dtype, kw, closure = render_text(row)
        if len(text) < 20:
            continue
        chunks.append({"chunk_id": str(uuid.uuid4()), "text": text, "doctype": dtype,
                       "kw": kw, "closure": closure, "domain": _g(row, "노선명") or "고속도로"})
    print(f"[render] 유효 청크 {len(chunks)} (예시: {chunks[0]['text'][:90]}...)", flush=True)

    # 임베딩 (배치)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(SBERT_MODEL)
    texts = [c["text"] for c in chunks]
    embs = model.encode(texts, batch_size=args.batch, normalize_embeddings=True,
                        show_progress_bar=False)
    dim = int(embs.shape[1])
    print(f"[embed] {SBERT_MODEL} → {embs.shape}", flush=True)

    # 적재
    out_path = Path(args.out)
    if out_path.exists():
        out_path.unlink()
    out = duckdb.connect(str(out_path))
    init_rag_schema(out)
    doc_id = str(uuid.uuid4())
    out.execute("INSERT INTO documents VALUES (?,?,?,?,?)",
                [doc_id, Path(args.src).name, "highway", "사고차단기록", "exkor_lockdown"])
    out.executemany(
        "INSERT INTO chunks VALUES (?,?,?,?,?,?,?)",
        [(c["chunk_id"], doc_id, c["text"], json.dumps(c["kw"], ensure_ascii=False),
          c["domain"], c["doctype"], c["closure"]) for c in chunks])
    out.executemany(
        "INSERT INTO chunk_embeddings VALUES (?,?,?,?,?,?)",
        [(str(uuid.uuid4()), c["chunk_id"], SBERT_MODEL, dim,
          json.dumps(embs[i].astype(float).tolist()), True)
         for i, c in enumerate(chunks)])
    n_chunks = out.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    n_emb = out.execute("SELECT COUNT(*) FROM chunk_embeddings").fetchone()[0]
    n_closure = out.execute("SELECT COUNT(*) FROM chunks WHERE closure_min IS NOT NULL").fetchone()[0]
    out.close()
    print(f"[save] {out_path}: 청크 {n_chunks}, 임베딩 {n_emb} ({SBERT_MODEL} {dim}d), "
          f"차단시간보유 {n_closure}", flush=True)
    print(f"\n✅ 벡터 RAG DB 생성 완료 → RAG_DB_PATH={out_path}", flush=True)


if __name__ == "__main__":
    main()
