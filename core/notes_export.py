# -*- coding: utf-8 -*-
"""
회의록 Markdown 저장/HTML 저장/간이 공유 서버
"""
from pathlib import Path
from typing import Tuple
import os, threading, socket, webbrowser
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

def save_markdown(md_text: str, path: str) -> str:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(md_text, encoding="utf-8")
    return str(p.resolve())

HTML_TEMPLATE = """<!doctype html><html lang="ko"><meta charset="utf-8">
<title>{title}</title>
<style>body{{font-family: -apple-system,Segoe UI,Roboto,Pretendard,Apple SD Gothic Neo,sans-serif;max-width:900px;margin:40px auto;line-height:1.6}}
code,pre{{background:#f5f7fa;padding:.2em .4em;border-radius:6px}}
a.btn{{display:inline-block;padding:.6em 1em;margin-right:.5em;border-radius:8px;background:#3b82f6;color:#fff;text-decoration:none}}
hr{{border:none;border-top:1px solid #eee;margin:24px 0}}</style>
<body>
<div>
<a class="btn" href="javascript:window.print()">🖨️ 인쇄/저장</a>
<a class="btn" onclick="copy()">📋 복사</a>
<script>function copy(){navigator.clipboard.writeText(document.body.innerText);alert('본문이 복사되었습니다');}</script>
</div>
<hr>
<article>{body}</article>
</body></html>
"""

def markdown_to_html(md_text: str) -> str:
    # 의존성 없이 매우 단순 변환: #, ##, -, **bold**, `code`
    import html, re
    txt = html.escape(md_text)
    txt = re.sub(r"^# (.*)$", r"<h1>\1</h1>", txt, flags=re.M)
    txt = re.sub(r"^## (.*)$", r"<h2>\1</h2>", txt, flags=re.M)
    txt = re.sub(r"^- (.*)$", r"<li>\1</li>", txt, flags=re.M)
    txt = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", txt)
    txt = re.sub(r"`(.+?)`", r"<code>\1</code>", txt)
    # 리스트 묶기
    lines = txt.split("\n")
    out, in_ul = [], False
    for L in lines:
        if L.startswith("<li>"):
            if not in_ul: out.append("<ul>"); in_ul=True
            out.append(L)
        else:
            if in_ul: out.append("</ul>"); in_ul=False
            out.append(f"<p>{L}</p>" if not L.startswith("<h") else L)
    if in_ul: out.append("</ul>")
    return "\n".join(out)

def save_html(md_text: str, path: str, title: str="회의록") -> str:
    body = markdown_to_html(md_text)
    html = HTML_TEMPLATE.format(title=title, body=body)
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")
    return str(p.resolve())

# ------------- 간이 공유 서버 --------------
_server_lock = threading.Lock()
_server_thread = None
_base_dir = None
_port = None

def _find_free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); addr, port = s.getsockname(); s.close()
    return port

def start_share_server(base_dir: str) -> Tuple[str, int]:
    """base_dir를 루트로 로컬 서버 기동. (이미 실행되어 있으면 재사용)"""
    global _server_thread, _base_dir, _port
    with _server_lock:
        if _server_thread and _server_thread.is_alive() and _base_dir == base_dir:
            return f"http://127.0.0.1:{_port}", _port
        _base_dir = base_dir
        _port = _find_free_port()
        os.chdir(base_dir)

        class Handler(SimpleHTTPRequestHandler):
            def log_message(self, fmt, *args):  # 조용히
                pass

        def _run():
            httpd = ThreadingHTTPServer(("127.0.0.1", _port), Handler)
            httpd.serve_forever()

        _server_thread = threading.Thread(target=_run, daemon=True)
        _server_thread.start()
        return f"http://127.0.0.1:{_port}", _port
