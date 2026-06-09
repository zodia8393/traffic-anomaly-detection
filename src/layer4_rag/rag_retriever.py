"""외부 RAG 지식베이스 리트리버 (DB 파일 플러그인, 고성능).

협업자가 구축한 RAG DB(사고/차단시간 지식, JHJ RAG 스키마)를 **파일로 받아 즉시 결합**한다.
우리 사고보고서 생성(report_generator)의 similar_cases 주입점에 단방향 보조지식으로 연결.

성능: 임베딩 행렬을 1회 로드→정규화→np.float32 캐시(TTL, atomic swap). 쿼리는 단일 matmul
q@M.T 벡터화. 융합은 RRF(스케일 무관). (JHJ의 매쿼리 전체로드+파이썬 선형 cosine 대비 ~60배)

견고성(적대적검증 27건 반영):
- 관련도 게이트: 키워드/벡터 신호가 전혀 없는 청크는 반환 금지(무관 사례 주입 방지).
- 단일 정렬 쿼리(chunks LEFT JOIN documents/embeddings)로 텍스트-임베딩 1:1 정렬 보장
  (id 컬럼 부재/비유니크 시에도 임베딩이 단일청크로 붕괴하지 않음).
- 임베딩 다중모델/구버전은 현행(is_current)·최빈모델로 필터.
- 쿼리 임베더는 DB 임베딩 모델과 정확히 일치할 때만 벡터검색(차원우연 cross-space 방지).
- DB 파일 없으면 무동작. 싱글톤은 경로·mtime 키라 DB 늦은 도착·교체에 자동 재연결.
- 손상/NaN 임베딩 개별 skip, OOM 상한, 광범위 except에 경고로그.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import json
import logging
import os
import re
import threading
import time
from typing import Any, Callable

import numpy as np

from config_new import (
    RAG_CACHE_TTL_SEC, RAG_CHUNKS_TABLE, RAG_DB_PATH, RAG_DOCS_TABLE,
    RAG_EMBED_MODEL, RAG_EMBED_PROVIDER, RAG_EMBED_TABLE, RAG_MAX_CHUNKS,
    RAG_OLLAMA_URL, RAG_REQUIRE_EXACT_MODEL, RAG_RRF_K, RAG_SIM_FLOOR, RAG_TOP_K,
)

logger = logging.getLogger(__name__)
_SBERT_IDS = ("minilm", "mpnet", "bge", "e5", "gte", "ko-sbert", "sentence-transformers", "/")


class RagKnowledgeRetriever:
    """외부 RAG DB 파일에서 하이브리드(벡터+키워드) 검색. 파일 없으면 무동작."""

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or RAG_DB_PATH
        self._conn = None
        self._available = False
        # 캐시 (atomic swap 대상)
        self._cache_ts = 0.0
        self._mat: np.ndarray | None = None
        self._vec_mask: np.ndarray | None = None
        self._meta: list[dict] = []
        self._kw_index: list[set] = []
        self._embed_model_name: str | None = None
        self._query_embed: Callable[[str], np.ndarray] | None = None
        self._schema: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._connect()

    # ── 연결/스키마 감지 ─────────────────────────────────────────────
    def _connect(self) -> None:
        if not self.db_path or not _Path(self.db_path).exists():
            logger.info("RAG DB 미존재(%s) → RAG 무동작", self.db_path)
            return
        try:
            import duckdb
            self._conn = duckdb.connect(self.db_path, read_only=True)
            self._detect_schema()
            if self._schema.get("chunks_ok"):
                self._available = True
                logger.info("RAG DB 연결: %s (벡터=%s, docs조인=%s)", self.db_path,
                            self._schema.get("has_vectors"), self._schema.get("docs_join"))
            else:
                logger.warning("RAG DB에 유효 청크 텍스트 컬럼 미발견(table=%s) → 무동작", RAG_CHUNKS_TABLE)
        except Exception as e:  # noqa: BLE001
            logger.error("RAG DB 연결 실패(%s): %s", self.db_path, e)
            self._conn = None

    def _cols(self, table: str) -> list[str]:
        try:
            return [r[0] for r in self._conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name=?",
                [table]).fetchall()]
        except Exception:  # noqa: BLE001
            return []

    def _detect_schema(self) -> None:
        cc = self._cols(RAG_CHUNKS_TABLE)
        s: dict[str, Any] = {"ccols": cc}
        s["text_col"] = _first(cc, ("chunk_text", "text", "content"))
        s["chunks_ok"] = bool(cc) and s["text_col"] is not None
        s["id_col"] = _first(cc, ("chunk_id", "id"))
        s["kw_col"] = _first(cc, ("keywords_json", "keywords"))
        s["docid_col"] = _first(cc, ("document_id", "doc_id"))
        # domain/document_type: 청크 우선, 없으면 documents 조인
        dc = self._cols(RAG_DOCS_TABLE)
        s["docs_join"] = bool(dc) and s["docid_col"] is not None and \
            _first(dc, ("document_id", "doc_id")) is not None
        s["doc_id_col"] = _first(dc, ("document_id", "doc_id")) if s["docs_join"] else None
        s["domain_src"] = ("c" if "domain" in cc else ("d" if (s["docs_join"] and "domain" in dc) else None))
        s["doctype_src"] = ("c" if "document_type" in cc else ("d" if (s["docs_join"] and "document_type" in dc) else None))
        # 임베딩 위치
        ec = self._cols(RAG_EMBED_TABLE)
        emb_chunk = _first(cc, ("embedding_json", "embedding", "vector_json", "vector", "embedding_vector"))
        emb_etab = _first(ec, ("embedding_json", "embedding", "vector_json", "vector", "embedding_vector"))
        if emb_etab:
            s["emb_loc"], s["emb_col"] = "table", emb_etab
            s["emb_join"] = _first(ec, ("chunk_id", "id"))
            s["emb_model_col"] = _first(ec, ("embedding_model", "model", "model_name"))
            s["emb_current_col"] = _first(ec, ("is_current", "current", "active"))
        elif emb_chunk:
            s["emb_loc"], s["emb_col"] = "chunk", emb_chunk
            s["emb_join"] = None
            s["emb_model_col"] = _first(cc, ("embedding_model", "model"))
            s["emb_current_col"] = None
        else:
            s["emb_loc"] = s["emb_col"] = None
        # id 유니크성 (별도 임베딩테이블 조인 안전성)
        s["id_unique"] = self._is_unique(RAG_CHUNKS_TABLE, s["id_col"]) if s["id_col"] else False
        # 벡터 사용 가능 조건: 임베딩 존재 + (청크내장 or 유니크id 조인)
        s["has_vectors"] = bool(s["emb_col"]) and (s["emb_loc"] == "chunk" or s["id_unique"])
        if s["emb_col"] and not s["has_vectors"]:
            logger.warning("임베딩은 별도테이블이나 id(%s) 비유니크 → 벡터검색 비활성(키워드 전용)", s["id_col"])
        self._schema = s

    def _is_unique(self, table: str, col: str) -> bool:
        try:
            r = self._conn.execute(
                f"SELECT COUNT(*)=COUNT(DISTINCT {col}) AND COUNT({col})=COUNT(*) FROM {table}").fetchone()
            return bool(r[0])
        except Exception:  # noqa: BLE001
            return False

    @property
    def available(self) -> bool:
        return self._available

    # ── 캐시 적재 (단일 정렬 쿼리, atomic swap) ──────────────────────
    def _ensure_cache(self) -> None:
        if self._mat is not None and self._meta and (time.time() - self._cache_ts) < RAG_CACHE_TTL_SEC:
            return
        if not self._available:
            return
        with self._lock:
            if self._meta and (time.time() - self._cache_ts) < RAG_CACHE_TTL_SEC:
                return
            try:
                self._build_cache()
            except Exception as e:  # noqa: BLE001 — 기존 캐시 보존(atomic)
                logger.warning("RAG 캐시 적재 실패(기존 유지): %s", e)

    def _build_cache(self) -> None:
        s = self._schema
        # OOM 상한
        try:
            n_total = self._conn.execute(f"SELECT COUNT(*) FROM {RAG_CHUNKS_TABLE}").fetchone()[0]
        except Exception:  # noqa: BLE001
            n_total = 0
        limit = ""
        if n_total > RAG_MAX_CHUNKS:
            logger.warning("청크 %d > 상한 %d → 상위 %d만 캐시(나머지 제외)", n_total, RAG_MAX_CHUNKS, RAG_MAX_CHUNKS)
            limit = f" LIMIT {RAG_MAX_CHUNKS}"

        # 현행/최빈 임베딩 모델 결정 (다중모델 오염 방지)
        model = RAG_EMBED_MODEL or self._resolve_model()

        # 단일 정렬 쿼리: chunks(c) LEFT JOIN documents(d) [LEFT JOIN embeddings(e)]
        sel = [f"c.{s['text_col']} AS _text",
               f"c.{s['id_col']}" if s["id_col"] else "NULL", f" AS _cid"]
        cols = [f"c.{s['text_col']} AS _text",
                (f"c.{s['id_col']}" if s["id_col"] else "NULL") + " AS _cid",
                (f"c.{s['kw_col']}" if s["kw_col"] else "NULL") + " AS _kw"]
        # domain/doctype 소스
        cols.append((f"c.domain" if s["domain_src"] == "c" else (f"d.domain" if s["domain_src"] == "d" else "NULL")) + " AS _domain")
        cols.append((f"c.document_type" if s["doctype_src"] == "c" else (f"d.document_type" if s["doctype_src"] == "d" else "NULL")) + " AS _doctype")
        # 임베딩
        emb_expr = "NULL"
        if s["has_vectors"]:
            if s["emb_loc"] == "chunk":
                emb_expr = f"c.{s['emb_col']}"
            else:
                emb_expr = f"e.{s['emb_col']}"
        cols.append(emb_expr + " AS _emb")

        joins = ""
        if s["docs_join"]:
            joins += f" LEFT JOIN {RAG_DOCS_TABLE} d ON c.{s['docid_col']} = d.{s['doc_id_col']}"
        if s["has_vectors"] and s["emb_loc"] == "table":
            cond = [f"c.{s['id_col']} = e.{s['emb_join']}"]
            if s["emb_model_col"] and model is not None:
                cond.append(f"e.{s['emb_model_col']} = ?")
            if s["emb_current_col"]:
                cond.append(f"e.{s['emb_current_col']}")
            joins += f" LEFT JOIN {RAG_EMBED_TABLE} e ON " + " AND ".join(cond)

        params = [model] if (s["has_vectors"] and s["emb_loc"] == "table" and s["emb_model_col"] and model is not None) else []
        q = f"SELECT {', '.join(cols)} FROM {RAG_CHUNKS_TABLE} c{joins}{limit}"
        rows = self._conn.execute(q, params).fetchall()

        meta, kw_index, vec_rows, dim = [], [], [], None
        for r in rows:
            text, cid, kw, dom, dtype, emb = r[0] or "", r[1], r[2], r[3], r[4], r[5]
            meta.append({"chunk_id": cid, "chunk_text": text,
                         "keywords": _parse_keywords(kw), "domain": dom, "document_type": dtype})
            kw_index.append(_tokenize(text))
            v = _parse_vector(emb) if s["has_vectors"] else None
            if v is not None:
                if dim is None:
                    dim = len(v)
                elif len(v) != dim:
                    v = None
            vec_rows.append(v)

        # 임베딩 행렬 (위치정렬 — id 불필요, 붕괴 없음)
        mat = mask = None
        if dim:
            mat = np.zeros((len(meta), dim), dtype=np.float32)
            mask = np.zeros(len(meta), dtype=bool)
            for i, v in enumerate(vec_rows):
                if v is None:
                    continue
                nrm = np.linalg.norm(v)
                mat[i] = v / nrm if nrm > 0 else v
                mask[i] = nrm > 0
        # 쿼리 임베더 (모델 정확매칭)
        embed_fn = self._make_query_embedder(model) if (mat is not None and mask is not None and mask.any()) else None
        if mat is not None and embed_fn is None:
            logger.warning("쿼리 임베더 미구성(model=%s, provider=%s, exact=%s) → 키워드 전용",
                           model, RAG_EMBED_PROVIDER, RAG_REQUIRE_EXACT_MODEL)

        # atomic swap
        self._meta, self._kw_index = meta, kw_index
        self._mat, self._vec_mask = mat, mask
        self._embed_model_name, self._query_embed = model, embed_fn
        self._cache_ts = time.time()
        logger.info("RAG 캐시: 청크 %d, 벡터 %s, 임베더 %s", len(meta),
                    None if mat is None else mat.shape, "ON" if embed_fn else "OFF(키워드)")

    def _resolve_model(self) -> str | None:
        s = self._schema
        if not s.get("emb_model_col") or s["emb_loc"] != "table":
            return None
        try:
            r = self._conn.execute(
                f"SELECT {s['emb_model_col']} FROM {RAG_EMBED_TABLE} "
                f"WHERE {s['emb_model_col']} IS NOT NULL "
                f"GROUP BY {s['emb_model_col']} ORDER BY COUNT(*) DESC LIMIT 1").fetchone()
            return r[0] if r else None
        except Exception:  # noqa: BLE001
            return None

    # ── 쿼리 임베더 (DB 모델 정확매칭, 불가시 None=키워드강등) ────────
    def _make_query_embedder(self, model_name: str | None) -> Callable[[str], np.ndarray] | None:
        prov = RAG_EMBED_PROVIDER.lower()
        if prov == "none":
            return None
        # Ollama: DB 모델명을 그대로 사용 → exact by construction
        if prov in ("auto", "ollama") and model_name and (prov == "ollama" or self._ollama_alive()):
            safe = re.sub(r"[^A-Za-z0-9_\-:./]", "", str(model_name))[:128]
            if safe:
                def _ollama(text: str) -> np.ndarray:
                    import requests
                    r = requests.post(f"{RAG_OLLAMA_URL.rstrip('/')}/api/embeddings",
                                      json={"model": safe, "prompt": text}, timeout=(2, 8))
                    r.raise_for_status()
                    return np.asarray(r.json()["embedding"], dtype=np.float32)
                return _ollama
        # SBert: 모델명이 st-계열로 보이거나 미지정+exact완화 시에만
        if prov in ("auto", "sbert"):
            name = model_name or ""
            looks_st = any(k in name.lower() for k in _SBERT_IDS)
            if not name and not RAG_REQUIRE_EXACT_MODEL:
                name = "all-MiniLM-L6-v2"; looks_st = True
            if looks_st and name:
                try:
                    from sentence_transformers import SentenceTransformer
                    m = SentenceTransformer(name)
                    return lambda t: np.asarray(m.encode(t, normalize_embeddings=True), dtype=np.float32)
                except Exception:  # noqa: BLE001
                    return None
        return None

    def _ollama_alive(self) -> bool:
        try:
            import requests
            requests.get(f"{RAG_OLLAMA_URL.rstrip('/')}/api/tags", timeout=2).raise_for_status()
            return True
        except Exception:  # noqa: BLE001
            return False

    # ── 검색 (관련도 게이트 포함) ────────────────────────────────────
    def search(self, query_text: str, top_k: int | None = None,
               domains: list[str] | None = None,
               document_types: list[str] | None = None) -> list[dict]:
        if not self._available or not query_text or not query_text.strip():
            return []
        self._ensure_cache()
        n = len(self._meta)
        if n == 0:
            return []
        top_k = top_k or RAG_TOP_K
        cand = np.ones(n, dtype=bool)
        if domains:
            ds = set(domains); cand &= np.array([m["domain"] in ds for m in self._meta])
        if document_types:
            ts = set(document_types); cand &= np.array([m["document_type"] in ts for m in self._meta])
        if not cand.any():
            return []

        kw_scores = self._keyword_scores(query_text)         # 원점수(관련도 게이트용)
        kw_rank = _rank_desc(np.where(cand, kw_scores, -1.0))

        sims = None
        vec_rank = None
        if self._mat is not None and self._vec_mask is not None and self._query_embed is not None:
            try:
                q = self._query_embed(query_text)
                qn = np.linalg.norm(q)
                if qn > 0 and q.shape[0] == self._mat.shape[1]:
                    sims = self._mat @ (q / qn)
                    vec_rank = _rank_desc(np.where(cand & self._vec_mask, sims, -2.0))
            except Exception as e:  # noqa: BLE001
                logger.warning("쿼리 임베딩 실패 → 키워드만: %s", e)

        fused = np.zeros(n)
        for rank in (r for r in (vec_rank, kw_rank) if r is not None):
            fused += 1.0 / (RAG_RRF_K + rank + 1)

        order = np.argsort(fused)[::-1]
        out = []
        for idx in order:
            if not cand[idx]:
                continue
            # 관련도 게이트: 키워드/벡터 실제 신호가 있어야 통과(무관 사례 주입 방지)
            kw_ok = kw_scores[idx] > 0
            vec_ok = sims is not None and self._vec_mask[idx] and sims[idx] >= RAG_SIM_FLOOR
            if not (kw_ok or vec_ok):
                continue
            m = self._meta[idx]
            out.append({"chunk_id": m["chunk_id"], "chunk_text": m["chunk_text"],
                        "domain": m["domain"], "document_type": m["document_type"],
                        "score": round(float(fused[idx]), 5),
                        "excerpt": _excerpt(m["chunk_text"], query_text)})
            if len(out) >= top_k:
                break
        return out

    def _keyword_scores(self, query: str) -> np.ndarray:
        qtok = _tokenize(query)
        compact_q = re.sub(r"\s+", "", query.lower())
        scores = np.zeros(len(self._meta), dtype=np.float32)
        for i, (toks, meta) in enumerate(zip(self._kw_index, self._meta)):
            s = 0.0
            if qtok:
                s += len(qtok & toks) / len(qtok) * 0.7
            if compact_q and len(compact_q) >= 4 and compact_q in re.sub(r"\s+", "", meta["chunk_text"].lower()):
                s += 0.3
            ql = query.lower()
            for kw in meta["keywords"]:
                if kw and kw.lower() in ql:
                    s += 0.25
            scores[i] = min(s, 1.5)
        return scores

    # ── report_generator 호환 어댑터 ────────────────────────────────
    def search_similar(self, query_text: str | None = None, top_k: int = 3,
                       accident_type: str | None = None,
                       road_name: str | None = None) -> list[dict]:
        q = query_text or " ".join(str(x) for x in (accident_type, road_name) if x)
        hits = self.search(q, top_k=top_k)
        out = []
        for h in hits:
            out.append({
                "report_id": h.get("chunk_id"),
                "road_name": road_name or "",
                "direction": "",
                "accident_type": accident_type or h.get("document_type") or "",
                "cause": "",
                "vehicles_summary": (h.get("excerpt", "") or "")[:200],
                "score": h.get("score", 0.0),
                "source": "rag_kb",
            })
        return out

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass


# ── 모듈 유틸 ────────────────────────────────────────────────────────
def _first(cols: list[str], cands: tuple[str, ...]) -> str | None:
    return next((c for c in cands if c in cols), None)


def _parse_vector(raw: Any) -> np.ndarray | None:
    if raw is None:
        return None
    try:
        if isinstance(raw, (list, tuple)):
            v = np.asarray(raw, dtype=np.float32)
        elif isinstance(raw, (bytes, bytearray)):
            v = np.frombuffer(raw, dtype=np.float32).copy()
        elif isinstance(raw, str):
            v = np.asarray(json.loads(raw), dtype=np.float32)
        elif isinstance(raw, np.ndarray):
            v = raw.astype(np.float32)
        else:
            return None
    except Exception:  # noqa: BLE001
        return None
    if v.ndim != 1 or v.size == 0 or not np.all(np.isfinite(v)):
        return None
    return v


def _parse_keywords(raw: Any) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw]
    try:
        v = json.loads(raw)
        return [str(x) for x in v] if isinstance(v, list) else []
    except Exception:  # noqa: BLE001
        return [t for t in re.split(r"[,\s]+", str(raw)) if t]


_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣_]+")


def _tokenize(text: str) -> set:
    return {t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) >= 2}


def _rank_desc(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(scores))
    return rank


def _excerpt(text: str, query: str, max_len: int = 300) -> str:
    low = text.lower()
    idx = -1
    for t in _TOKEN_RE.findall(query):
        idx = low.find(t.lower())
        if idx >= 0:
            break
    start = max(0, idx - 80) if idx >= 0 else 0
    return text[start:start + max_len].replace("\n", " ").strip()


# 싱글톤 (경로·mtime 키 — DB 늦은 도착/교체 자동 재연결)
_retrievers: dict[str, tuple[RagKnowledgeRetriever, float | None]] = {}
_glock = threading.Lock()


def get_retriever(db_path: str | None = None) -> RagKnowledgeRetriever:
    path = db_path or RAG_DB_PATH
    mtime = os.path.getmtime(path) if os.path.exists(path) else None
    with _glock:
        cached = _retrievers.get(path)
        if cached is not None and cached[1] == mtime:
            return cached[0]
        r = RagKnowledgeRetriever(db_path=path)
        _retrievers[path] = (r, mtime)
        return r


def reset_retriever() -> None:
    with _glock:
        for r, _ in _retrievers.values():
            r.close()
        _retrievers.clear()
