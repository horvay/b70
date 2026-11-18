#!/usr/bin/env python3
"""A model server stand-in for test/standby.sh (no GPU): "loads" for a second, answers /v1/models, and answers a
completion after `delay` seconds (a field of the request), streamed with chunked transfer encoding and llama.cpp's
prompt_progress when asked. Logs every completion request (its tag and max_tokens) to the log file.

usage: standby_fake_server.py <port> <log>"""
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

port, log = int(sys.argv[1]), sys.argv[2]
time.sleep(1)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        pass

    def send_json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            self.send_json({"object": "list", "data": [{"id": "fake-model", "object": "model"}]})
        else:
            self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        with open(log, "a") as f:
            f.write(json.dumps({"t": time.time(), "tag": req.get("tag"), "max_tokens": req.get("max_tokens")}) + "\n")
        delay = float(req.get("delay", 0))
        if not req.get("stream"):
            time.sleep(delay)
            return self.send_json({"choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}}]})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def event(data):
            data = b"data: " + (data if isinstance(data, bytes) else json.dumps(data).encode()) + b"\n\n"
            self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
            self.wfile.flush()
        if req.get("return_progress"):
            event({"choices": [{"index": 0, "delta": {"role": "assistant", "content": None}}],
                   "prompt_progress": {"total": 10, "cache": 0, "processed": 10, "time_ms": 1}})
        time.sleep(delay)
        event({"choices": [{"index": 0, "delta": {"content": "hello"}}]})
        event(b"[DONE]")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
