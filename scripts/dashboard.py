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
from urllib.parse import urlparse, parse_qs

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
CKPT = ROOT / "checkpoints"
VENV_PY = ROOT / ".venv" / "bin" / "python"

CKPT.mkdir(exist_ok=True)

TRAIN_PATTERN = r"train(_selfplay(_paired)?|_equivariant|_league)?\.py --|selfplay_league_(loop\.sh|dynamic\.py)"
WATCH_PATTERN = r"watch\.py"
BREAKDOWN_KEYS = ["r_strike", "r_effort", "r_jerk", "r_engage", "r_down", "r_knockdown_entry", "r_balance", "r_ground", "r_knee", "r_terminal"]
ROLLOUT_KEYS = ["total_timesteps", "ep_rew_mean", "b_ep_rew_mean", "ep_len_mean", "fps"] + BREAKDOWN_KEYS
# metrics offered in the chart's series picker: total_timesteps is the x-axis, not a plottable series
CHART_METRICS = [k for k in ROLLOUT_KEYS if k not in ("total_timesteps", "fps")]


def pids_matching(pattern):
    out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    return [p for p in out.stdout.split() if p]


def latest_log():
    # exclude the dashboard's own stdout log and watch.py logs -- only training logs matter here
    logs = sorted(
        (p for p in CKPT.glob("*.log") if p.name != "dashboard.log" and not p.name.startswith("watch")),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
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


def parse_log_history(path, n_bytes=3_000_000, max_points=300):
    """Every rollout/ table in the log is one data point; only reads the last
    n_bytes of the file (these logs can grow into the tens of MB) and
    downsamples evenly so the response stays small regardless of run length."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - n_bytes))
            data = f.read().decode(errors="ignore")
    except OSError:
        return []
    points = []
    current = {}
    for line in data.splitlines():
        m = re.match(r"\|\s*(\w+)\s*\|\s*([-\d.eE]+)\s*\|", line)
        if m and m.group(1) in ROLLOUT_KEYS:
            current[m.group(1)] = float(m.group(2))
        elif line.startswith("---") and "total_timesteps" in current:
            points.append(current)
            current = {}
    if "total_timesteps" in current:
        points.append(current)
    if len(points) > max_points:
        stride = len(points) / max_points
        points = [points[int(i * stride)] for i in range(max_points)]
    return points


def _ckpt_role(name):
    # league checkpoints are named ppo_p1_*/ppo_p2_*[_bootstrap]_* -- P1 was only ever
    # trained controlling 'a' natively, P2 only ever controlling 'b' via the mirrored
    # pathway, and (per this session's root-cause finding) the two are NOT
    # interchangeable. Anything else (scratch/selfplay/vs_scripted/...) predates the
    # league split and was only ever trained playing 'a' -- fine for the A slot, but
    # untested/likely-weak in the B slot.
    tokens = re.split(r"[_/.]", name)
    if "p2" in tokens:
        return "P2"
    if "p1" in tokens:
        return "P1"
    return "공용"


def list_checkpoints():
    # selfplay_opponent_*.zip is an internal working copy the training loop overwrites
    # every refresh -- not a meaningful standalone checkpoint, so keep it out of the list
    entries = [(p, p.name, p.stat().st_mtime)
               for p in CKPT.glob("*.zip") if not p.name.startswith("selfplay_opponent_")]
    autosave = CKPT / "autosave"
    if autosave.exists():
        auto = sorted(autosave.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)[:15]
        entries += [(p, f"autosave/{p.name}", p.stat().st_mtime) for p in auto]
    # merged and sorted newest-first across both top-level and autosave, so index 0
    # is always the single most recent checkpoint regardless of which dir it's in
    entries.sort(key=lambda e: e[2], reverse=True)
    return [{"label": label, "path": str(p), "mtime": mtime, "role": _ckpt_role(label)}
            for p, label, mtime in entries]


def paginate_checkpoints(page, page_size, role=None):
    all_ckpts = list_checkpoints()
    if role and role != "all":
        all_ckpts = [c for c in all_ckpts if c["role"] == role]
    total = len(all_ckpts)
    page_size = min(max(page_size, 1), 200)
    total_pages = max(1, -(-total // page_size))  # ceil div
    page = min(max(page, 1), total_pages)
    start = (page - 1) * page_size
    return {
        "checkpoints": all_ckpts[start:start + page_size],
        "page": page, "page_size": page_size,
        "total": total, "total_pages": total_pages,
    }


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._serve_index()
        elif path == "/api/status":
            self._status()
        elif path == "/api/checkpoints":
            qs = parse_qs(urlparse(self.path).query)
            try:
                page = int(qs.get("page", ["1"])[0])
                page_size = int(qs.get("page_size", ["20"])[0])
            except ValueError:
                return self._json({"error": "invalid page/page_size"}, 400)
            role = qs.get("role", [None])[0]
            self._json(paginate_checkpoints(page, page_size, role))
        elif path == "/api/history":
            log = latest_log()
            self._json({
                "points": parse_log_history(log) if log else [],
                "available_metrics": CHART_METRICS,
            })
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
            "/api/matchup": self._matchup,
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

        selfplay = bool(payload.get("selfplay"))
        league = bool(payload.get("league"))
        league_dynamic = bool(payload.get("league_dynamic"))
        if league:
            if not resume_from:
                return self._json({"error": "league needs a checkpoint to bootstrap P1 and P2 from"}, 400)
            try:
                rounds = str(int(payload.get("rounds", 20)))
            except (TypeError, ValueError):
                return self._json({"error": "invalid rounds"}, 400)
            if league_dynamic:
                # selfplay_league_dynamic.py writes its own canonical log directly to
                # checkpoints/league_dynamic_loop.log regardless of what captures its
                # stdout -- use a different path here so the two writers don't collide
                # (matches the same hardcoded-internal-log pattern as the other loop
                # scripts; latest_log() will pick up whichever file is newest either way)
                log_path = CKPT / "league_dynamic_dashboard.log"
                try:
                    eval_episodes = str(int(payload.get("eval_episodes", 30)))
                    margin = str(int(payload.get("margin", 3)))
                except (TypeError, ValueError):
                    return self._json({"error": "invalid eval_episodes/margin"}, 400)
                cmd = [str(VENV_PY), "selfplay_league_dynamic.py",
                       "--rounds", rounds, "--timesteps", timesteps,
                       "--eval-episodes", eval_episodes, "--margin", margin,
                       "--p1-init", resume_from, "--p2-init", resume_from]
            else:
                log_path = CKPT / "league_loop_dashboard.log"
                cmd = ["bash", "selfplay_league_loop.sh", rounds, timesteps, resume_from, resume_from]
        elif selfplay:
            if not resume_from:
                return self._json({"error": "self-play needs a checkpoint to bootstrap from"}, 400)
            log_path = CKPT / "train_selfplay_dashboard.log"
            cmd = [str(VENV_PY), "train_selfplay.py", "--timesteps", timesteps,
                   "--out", out, "--init-from", resume_from]
            try:
                refresh = payload.get("refresh_interval")
                if refresh:
                    cmd += ["--refresh-interval", str(int(refresh))]
                ent_coef = payload.get("ent_coef")
                if ent_coef not in (None, ""):
                    cmd += ["--ent-coef", str(float(ent_coef))]
            except (TypeError, ValueError):
                return self._json({"error": "invalid refresh_interval/ent_coef"}, 400)
        else:
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
        if checkpoint == "__latest__":
            ckpts = list_checkpoints()
            checkpoint = ckpts[0]["path"] if ckpts else None
        if not checkpoint or not pathlib.Path(checkpoint).is_file():
            return self._json({"error": "valid checkpoint path required"}, 400)
        # 'opponent' is a separate checkpoint driving 'b' (e.g. a P2 league checkpoint) --
        # pass the same path as checkpoint for a self-play mirror match, or omit for the
        # scripted opponent. Using a single shared checkpoint for both sides only makes
        # sense for pre-league checkpoints; P1/P2 league checkpoints are NOT
        # interchangeable (see docs/기술문서 on the a/b asymmetry).
        opponent = payload.get("opponent") or None
        if opponent and not pathlib.Path(opponent).is_file():
            return self._json({"error": "opponent checkpoint not found"}, 400)
        time.sleep(0.3)
        log_path = CKPT / "watch_dashboard.log"
        env = dict(os.environ)
        env.setdefault("DISPLAY", ":0")
        cmd = [str(VENV_PY), "watch.py", checkpoint]
        if opponent:
            cmd += ["--opponent", opponent]
        with open(log_path, "wb") as logf:
            subprocess.Popen(
                cmd, cwd=str(SCRIPTS),
                stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True, env=env,
            )
        self._json({"started": True, "checkpoint": checkpoint, "opponent": opponent})

    def _watch_stop(self, _payload):
        pids = pids_matching(WATCH_PATTERN)
        for pid in pids:
            subprocess.run(["kill", pid])
        self._json({"stopped": pids})

    def _matchup(self, payload):
        ckpt_a = payload.get("checkpoint_a")
        ckpt_b = payload.get("checkpoint_b")
        if not ckpt_a or not pathlib.Path(ckpt_a).is_file():
            return self._json({"error": "checkpoint_a not found"}, 400)
        if not ckpt_b or not pathlib.Path(ckpt_b).is_file():
            return self._json({"error": "checkpoint_b not found"}, 400)
        try:
            episodes = min(max(int(payload.get("episodes", 60)), 5), 300)
        except (TypeError, ValueError):
            return self._json({"error": "invalid episodes"}, 400)

        try:
            out = subprocess.run(
                [str(VENV_PY), "matchup_eval.py", ckpt_a, ckpt_b, "--episodes", str(episodes)],
                cwd=str(SCRIPTS), capture_output=True, text=True, timeout=300,
            )
        except subprocess.TimeoutExpired:
            return self._json({"error": "matchup evaluation timed out (300s)"}, 504)
        if out.returncode != 0:
            return self._json({"error": "matchup evaluation failed", "detail": out.stderr[-2000:]}, 500)
        try:
            result = json.loads(out.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return self._json({"error": "could not parse matchup output", "detail": out.stdout[-2000:]}, 500)
        self._json(result)

    def log_message(self, fmt, *args):
        pass


def main():
    server = ThreadingHTTPServer(("0.0.0.0", 8787), Handler)
    print("fightai dashboard: http://localhost:8787")
    server.serve_forever()


if __name__ == "__main__":
    main()
