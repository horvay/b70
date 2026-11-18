#!/usr/bin/env python3
# A small chat page for an OpenAI-compatible server (`b70 serve`): serves chat.html (+ chat.js, prompt.js) and forwards
# /v1/* to the server, so the browser needs no CORS. Nothing is logged here or saved server-side: chats live in the
# browser's localStorage (and in files you export). Standard library only, no GPU.
#
#   chat.py [--listen 127.0.0.1:8131] [--server http://127.0.0.1:8130]
import argparse, http.client, os, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"   # the body ends when the connection closes, which suits a streamed answer

    def log_message(self, *args):
        pass

    def do_GET(self):
        page = {"/": "chat.html", "/index.html": "chat.html", "/chat.js": "chat.js", "/prompt.js": "prompt.js"}.get(self.path)
        if page:
            with open(os.path.join(HERE, page), "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8" if page.endswith(".html") else "text/javascript; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/v1/"):
            self.forward(None)
        else:
            self.send_error(404)

    def do_POST(self):
        if not self.path.startswith("/v1/"):
            return self.send_error(404)
        self.forward(self.rfile.read(int(self.headers.get("Content-Length") or 0)))

    def forward(self, body):
        u = urllib.parse.urlparse(self.server.upstream)
        conn = http.client.HTTPConnection(u.hostname, u.port, timeout=3600)
        try:
            conn.request(self.command, self.path, body, {"Content-Type": "application/json"})
            r = conn.getresponse()
            self.send_response(r.status)
            self.send_header("Content-Type", r.getheader("Content-Type") or "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            while True:
                chunk = r.read1(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass   # the page stopped the answer; closing the upstream connection stops the generation
        except OSError as ex:
            try:
                self.send_error(502, f"server not reachable: {ex}")
            except OSError:
                pass
        finally:
            conn.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--listen", default="127.0.0.1:8131")
    p.add_argument("--server", default="http://127.0.0.1:8130")
    a = p.parse_args()
    host, port = a.listen.rsplit(":", 1)
    srv = ThreadingHTTPServer((host, int(port)), Handler)
    srv.upstream = a.server
    srv.daemon_threads = True
    print(f"chat on http://{a.listen}/  ->  {a.server}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
