"""Local chat UI for trying the pipeline in a browser. Standard library only.

Usage: python -m nlp_proto.web [port]   then open http://localhost:8765
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .pipeline import Assistant

STATIC = os.path.join(os.path.dirname(__file__), "..", "web", "index.html")
assistant = None


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ("/", "/index.html"):
            self.send_error(404)
            return
        with open(STATIC, "rb") as fh:
            body = fh.read()
        self._send(200, body, "text/html; charset=utf-8")

    def do_POST(self):
        if self.path != "/api/message":
            self.send_error(404)
            return
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if req.get("draft"):
                out = assistant.follow_up(req["draft"], text=req.get("text"), category=req.get("category"))
            else:
                out = assistant.handle(req.get("text", ""))
            self._send(200, json.dumps(out, ensure_ascii=False, default=str).encode("utf-8"), "application/json")
        except Exception as ex:  # keep the demo alive and show the error in the page
            self._send(500, json.dumps({"error": str(ex)}).encode("utf-8"), "application/json")

    def _send(self, code, body, content_type):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass


def main():
    global assistant
    port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("PORT", 8765))
    assistant = Assistant()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Chat UI running on http://localhost:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
