"""사고 차단시간 RAG 챗봇 웹 서버 (stdlib, 의존성 0).

브라우저에서 보면서 질문하는 채팅 UI. 벡터 RAG(rag_knowledge.duckdb) 위에서 유사 실사고
회수 + 실제 차단시간 통계로 답한다.

실행:
  python serve_chatbot.py            # http://localhost:8765
  python serve_chatbot.py --port 9000 --llm
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from layer4_rag.chatbot import RagChatbot

_BOT: RagChatbot | None = None

PAGE = """<!doctype html><html lang=ko><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>사고 차단시간 RAG 챗봇</title><style>
:root{--bg:#0f1419;--panel:#1a212b;--accent:#3b9eff;--mut:#8b97a7;--ok:#2ec27e}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:#e6edf3;font-family:'Apple SD Gothic Neo',sans-serif}
header{padding:14px 20px;background:#0b2540;border-bottom:1px solid #1f3a5f;display:flex;align-items:center;gap:10px}
header b{font-size:17px}header span{color:var(--mut);font-size:13px}
#chat{max-width:860px;margin:0 auto;padding:18px 16px 120px}
.msg{margin:14px 0;display:flex}.msg.u{justify-content:flex-end}
.bub{max-width:80%;padding:12px 15px;border-radius:14px;line-height:1.55;white-space:pre-wrap;font-size:14.5px}
.u .bub{background:var(--accent);color:#fff;border-bottom-right-radius:4px}
.b .bub{background:var(--panel);border:1px solid #28323f;border-bottom-left-radius:4px}
.stats{display:flex;gap:14px;margin:8px 0 4px;flex-wrap:wrap}
.stat{background:#13202e;border:1px solid #234;border-radius:10px;padding:8px 12px;text-align:center}
.stat b{display:block;font-size:20px;color:var(--ok)}.stat span{font-size:11px;color:var(--mut)}
.case{margin:6px 0;padding:9px 11px;background:#10171f;border-left:3px solid var(--accent);border-radius:6px;font-size:13px;color:#c4d0dc}
.case .sc{color:var(--mut);font-size:11px;float:right}
form{position:fixed;bottom:0;left:0;right:0;background:#0b1119;border-top:1px solid #1f2a36;padding:12px}
.row{max-width:860px;margin:0 auto;display:flex;gap:8px}
input{flex:1;padding:13px 15px;border-radius:12px;border:1px solid #2a3744;background:#121a23;color:#e6edf3;font-size:15px;outline:none}
button{padding:0 22px;border:0;border-radius:12px;background:var(--accent);color:#fff;font-size:15px;font-weight:600;cursor:pointer}
button:disabled{opacity:.5}.chk{display:flex;align-items:center;gap:5px;color:var(--mut);font-size:12px;max-width:860px;margin:6px auto 0}
.load{color:var(--mut);font-style:italic}
</style></head><body>
<header><b>🤖 사고 차단시간 RAG</b><span id=meta>연결 중…</span></header>
<div id=chat></div>
<form id=f><div class=row><input id=q placeholder="예: 경부선 화물차 전복 적재물유출 차단시간은?" autocomplete=off autofocus>
<button id=send>전송</button></div>
<label class=chk><input type=checkbox id=llm checked> AI 종합답변(상시 로드 · 끄면 빠른 검색)</label></form>
<script>
const chat=document.getElementById('chat'),f=document.getElementById('f'),q=document.getElementById('q'),send=document.getElementById('send');
fetch('/health').then(r=>r.json()).then(d=>{document.getElementById('meta').textContent=d.ok?`지식 ${d.chunks}건 · ${d.model} · LLM ${d.llm}`:'RAG 미연결'});
function add(t,cls){const m=document.createElement('div');m.className='msg '+cls;const b=document.createElement('div');b.className='bub';if(typeof t=='string')b.textContent=t;else b.appendChild(t);m.appendChild(b);chat.appendChild(m);window.scrollTo(0,9e9);return b;}
function render(d){const w=document.createElement('div');const a=document.createElement('div');a.textContent=d.answer;w.appendChild(a);
if(d.stats){const s=document.createElement('div');s.className='stats';for(const[k,lbl]of[['avg','평균(분)'],['median','중앙값'],['min','최소'],['max','최대'],['n','사례수']]){const c=document.createElement('div');c.className='stat';c.innerHTML=`<b>${d.stats[k]}</b><span>${lbl}</span>`;s.appendChild(c);}w.appendChild(s);}
(d.cases||[]).forEach(c=>{const e=document.createElement('div');e.className='case';e.innerHTML=`<span class=sc>${(c.score*100).toFixed(1)}</span>${c.text.replace(/</g,'&lt;')}`;w.appendChild(e);});return w;}
f.onsubmit=async ev=>{ev.preventDefault();const text=q.value.trim();if(!text)return;add(text,'u');q.value='';send.disabled=true;
const l=add('검색 중…','b');l.className='bub load';
try{const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:text,llm:document.getElementById('llm').checked})});
const d=await r.json();l.parentElement.remove();add(render(d),'b');}catch(e){l.textContent='오류: '+e;}send.disabled=false;q.focus();};
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # 조용히
        pass

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.end_headers()
        self.wfile.write(body.encode("utf-8") if isinstance(body, str) else body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html")
        elif self.path == "/health":
            r = _BOT.r
            self._send(200, json.dumps({"ok": _BOT.available, "chunks": len(r._meta) if r._meta else 0,
                                        "model": r._embed_model_name or "키워드",
                                        "llm": "상시" if _BOT._client is not None else "off"}))
        else:
            self._send(404, "{}")

    def do_POST(self):
        if self.path != "/ask":
            self._send(404, "{}"); return
        try:
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n) or b"{}")
            _BOT.use_llm = bool(data.get("llm"))
            out = _BOT.ask_structured(str(data.get("question", "")).strip(), top_k=5)
            self._send(200, json.dumps(out, ensure_ascii=False))
        except Exception:  # noqa: BLE001
            self._send(500, json.dumps({"answer": "서버 오류: " + traceback.format_exc()[-300:],
                                        "stats": None, "cases": []}, ensure_ascii=False))


def main():
    global _BOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--fast", action="store_true", help="LLM 비활성(검색+통계만, 경량)")
    args = ap.parse_args()

    print("RAG 챗봇 로딩(임베더)…", flush=True)
    _BOT = RagChatbot(use_llm=not args.fast)
    if not _BOT.available:
        print("⚠️ RAG DB(rag_knowledge.duckdb) 미연결 — RAG_DB_PATH 확인", flush=True)
    else:
        _BOT.r._ensure_cache()
        print(f"✅ RAG 준비: 청크 {len(_BOT.r._meta)}, 임베더 {_BOT.r._embed_model_name}", flush=True)
    if not args.fast:
        print("LLM(Qwen) 사전로딩 + 워밍업…", flush=True)
        t = time.time()
        from layer3_mllm.mllm_client import MLLMClient
        _BOT._client = MLLMClient(backend="transformers")
        try:  # 워밍업 추론 — 가중치 폴트인해 첫 사용자 질문도 빠르게
            _BOT._client.chat([{"role": "user", "content": "안녕"}], max_tokens=4, parse_json=False)
        except Exception:  # noqa: BLE001
            pass
        print(f"✅ LLM 상시대기·워밍업 완료 ({time.time()-t:.0f}초)", flush=True)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"🌐 http://localhost:{args.port}  (Ctrl+C 종료)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
