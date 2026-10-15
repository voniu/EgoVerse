"""Local page for clicking the two finger pads of the gripper (standard library only).

Serves the labelling page and <workdir>/frames.json, <workdir>/frames/*.jpg, and keeps
labels in <workdir>/labels.json. Binds to 127.0.0.1 only.

    python -m egomimic.scripts.piper_process.tcp_calibration.label_server \
        --workdir <tcp_workdir> --port 8765
"""

import argparse
import json
import os
import tempfile
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = Path(__file__).with_name("index.html")


def make_handler(workdir):
    labels = workdir / "labels.json"

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(workdir), **kwargs)

        def _send(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status, payload):
            self._send(status, json.dumps(payload).encode(), "application/json")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            if self.path == "/labels":
                return self._json(
                    200, json.loads(labels.read_text()) if labels.exists() else {}
                )
            return super().do_GET()

        def do_POST(self):
            if self.path != "/labels":
                return self._json(404, {"error": "not found"})
            length = int(self.headers.get("Content-Length", 0))
            try:
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError
            except ValueError:
                return self._json(400, {"error": "expected a JSON object"})
            fd, tmp = tempfile.mkstemp(dir=workdir, suffix=".tmp")
            with os.fdopen(fd, "w") as handle:
                json.dump(data, handle, indent=1)
            os.replace(tmp, labels)
            return self._json(200, {"saved": len(data)})

        def log_message(self, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    workdir = args.workdir.resolve()
    if not (workdir / "frames.json").exists():
        raise SystemExit(f"{workdir}/frames.json missing; run select_frames first")
    print(
        f"labelling at http://127.0.0.1:{args.port}, labels -> {workdir}/labels.json",
        flush=True,
    )
    ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(workdir)).serve_forever()


if __name__ == "__main__":
    main()
