#!/usr/bin/env python3
"""
Thin HTTP proxy for llama-server — patches the missing GET /v1/models/{id}
endpoint that Claude Code uses to validate the model before sending messages.

Usage:
    python3 llama-proxy.py --target http://127.0.0.1:10222 --port 10220 --model gemma-4-E2B-it-Q4_K_M.gguf
"""
import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer


def make_handler(target: str, model_id: str):
    class ProxyHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            # Intercept GET /v1/models/<id> — llama-server returns 404 for this
            if re.match(r"^/v1/models/[^/]+$", self.path):
                payload = json.dumps({
                    "type": "model",
                    "id": model_id,
                    "display_name": model_id,
                    "created_at": "2026-04-01T00:00:00Z",
                }).encode()
                self._respond(200, {"Content-Type": "application/json"}, payload)
            else:
                self._proxy()

        def do_POST(self):
            self._proxy()

        def do_DELETE(self):
            self._proxy()

        def _proxy(self):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else None
            url = target + self.path
            req = urllib.request.Request(url, data=body, method=self.command)
            for k, v in self.headers.items():
                if k.lower() not in ("host", "content-length", "transfer-encoding"):
                    req.add_header(k, v)
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    data = r.read()
                    headers = {k: v for k, v in r.headers.items()
                               if k.lower() not in ("transfer-encoding",)}
                    self._respond(r.status, headers, data)
            except urllib.error.HTTPError as e:
                data = e.read()
                self._respond(e.code, {}, data)

        def _respond(self, status, headers, body):
            self.send_response(status)
            for k, v in headers.items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            pass  # silence per-request logs

    return ProxyHandler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="http://127.0.0.1:10222")
    parser.add_argument("--port", type=int, default=10220)
    parser.add_argument("--model", default="gemma-4-E2B-it-Q4_K_M.gguf")
    args = parser.parse_args()

    server = HTTPServer(("0.0.0.0", args.port), make_handler(args.target, args.model))
    print(f"Proxy: 0.0.0.0:{args.port} → {args.target}  (model={args.model})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Proxy stopped.")
