"""Local dashboard: monitor the training/watch processes and trigger new runs.

Run with: .venv/bin/python scripts/dashboard.py
Then open http://localhost:8787
"""
import json
import os
import pathlib
import re
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
CKPT = ROOT / "checkpoints"
VENV_PY = ROOT / ".venv" / "bin" / "python"

CKPT.mkdir(exist_ok=True)

TRAIN_PATTERN = "train.py --timesteps"
WATCH_PATTERN = "watch.py"
ROLLOUT_KEYS = ["total_timesteps", "ep_rew_mean", "ep_len_mean", "fps"]


def pids_matching(pattern):
    out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    return [p for p in out.stdout.split() if p]


def latest_log():
    logs = sorted(CKPT.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def parse_log_tail(path, n_bytes=12_000):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - n_bytes))
            data = f.read().decode(errors="ignore")
    except OSError:
        return {}
    metrics = {}
    for line in reversed(data.splitlines()):
        m = re.match(r"\|\s*(\w+)\s*\|\s*([-\d.eE]+)\s*\|", line)
        if m and m.group(1) in ROLLOUT_KEYS and m.group(1) not in metrics:
            metrics[m.group(1)] = float(m.group(2))
        if len(metrics) == len(ROLLOUT_KEYS):
            break
    return metrics


def list_checkpoints():
    result = []
    for p in sorted(CKPT.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True):
        result.append({"label": p.name, "path": str(p), "mtime": p.stat().st_mtime})
    autosave = CKPT / "autosave"
    if autosave.exists():
        auto = sorted(autosave.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)[:15]
        for p in auto:
            result.append({"label": f"autosave/{p.name}", "path": str(p), "mtime": p.stat().st_mtime})
    return result


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self._serve_index()
        elif self.path == "/api/status":
            self._status()
        elif self.path == "/api/checkpoints":
            self._json({"checkpoints": list_checkpoints()})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw or b"{}")
        except ValueError:
            payload = {}
        routes = {
            "/api/train/start": self._train_start,
            "/api/train/stop": self._train_stop,
            "/api/watch/start": self._watch_start,
            "/api/watch/stop": self._watch_stop,
        }
        fn = routes.get(self.path)
        if fn is None:
            self.send_response(404)
            self.end_headers()
            return
        fn(payload)

    def _serve_index(self):
        html = (SCRIPTS / "dashboard_index.html").read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def _status(self):
        train_pids = pids_matching(TRAIN_PATTERN)
        watch_pids = pids_matching(WATCH_PATTERN)
        log = latest_log()
        self._json({
            "training": bool(train_pids),
            "watching": bool(watch_pids),
            "log_file": log.name if log else None,
            "metrics": parse_log_tail(log) if log else {},
        })

    def _train_start(self, payload):
        if pids_matching(TRAIN_PATTERN):
            return self._json({"error": "training already running"}, 409)
        try:
            timesteps = str(int(payload.get("timesteps", 2_000_000)))
        except (TypeError, ValueError):
            return self._json({"error": "invalid timesteps"}, 400)
        out = (payload.get("out") or f"dash_{int(time.time())}").strip()
        if not re.fullmatch(r"[\w.-]+", out):
            return self._json({"error": "invalid out name"}, 400)
        resume_from = payload.get("resume_from") or None
        if resume_from and not pathlib.Path(resume_from).is_file():
            return self._json({"error": "resume_from checkpoint not found"}, 400)

        log_path = CKPT / "train_dashboard.log"
        cmd = [str(VENV_PY), "train.py", "--timesteps", timesteps, "--out", out]
        if resume_from:
            cmd += ["--resume-from", resume_from]
        with open(log_path, "wb") as logf:
            subprocess.Popen(
                cmd, cwd=str(SCRIPTS), stdout=logf, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True,
            )
        self._json({"started": True, "out": out, "log_file": log_path.name})

    def _train_stop(self, _payload):
        pids = pids_matching(TRAIN_PATTERN)
        for pid in pids:
            subprocess.run(["kill", pid])
        self._json({"stopped": pids})

    def _watch_start(self, payload):
        for pid in pids_matching(WATCH_PATTERN):
            subprocess.run(["kill", pid])
        checkpoint = payload.get("checkpoint")
        if not checkpoint or not pathlib.Path(checkpoint).is_file():
            return self._json({"error": "valid checkpoint path required"}, 400)
        time.sleep(0.3)
        log_path = CKPT / "watch_dashboard.log"
        env = dict(os.environ)
        env.setdefault("DISPLAY", ":0")
        with open(log_path, "wb") as logf:
            subprocess.Popen(
                [str(VENV_PY), "watch.py", checkpoint], cwd=str(SCRIPTS),
                stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True, env=env,
            )
        self._json({"started": True})

    def _watch_stop(self, _payload):
        pids = pids_matching(WATCH_PATTERN)
        for pid in pids:
            subprocess.run(["kill", pid])
        self._json({"stopped": pids})

    def log_message(self, fmt, *args):
        pass


def main():
    server = ThreadingHTTPServer(("0.0.0.0", 8787), Handler)
    print("fightai dashboard: http://localhost:8787")
    server.serve_forever()


if __name__ == "__main__":
    main()
