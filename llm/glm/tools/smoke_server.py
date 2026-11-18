"""Queued, bounded server startup and long-prompt checks. Stops only the server it starts."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model")
    p.add_argument("--request", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--port", type=int, default=8140)
    p.add_argument("--ctx", type=int)
    p.add_argument("--turns", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--followup", action="store_true", help="append each reply and a new player action")
    p.add_argument("--rewrite-history", action="store_true", help="change early history each turn, forcing prefill after decode")
    p.add_argument("--cancel-after-prefill", action="store_true", help="disconnect after the first prefill chunk, then retry")
    a = p.parse_args()
    b70 = Path(__file__).resolve().parents[2] / "b70"
    output = Path(a.output)
    request = json.loads(Path(a.request).read_text())
    request["temperature"] = a.temperature
    url = f"http://127.0.0.1:{a.port}"
    # This program belongs inside b70q submit. The child inherits its ticket.
    import os
    if not os.environ.get("B70Q_TICKET"):
        p.error("run this check through b70q submit or b70-lease run")
    with output.with_suffix(".server.log").open("w") as log:
        command = [str(b70), "serve", a.model, "--port", str(a.port)]
        if a.ctx:
            command += ["--ctx", str(a.ctx)]
        proc = subprocess.Popen(command,
                                stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 900
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"server exited {proc.returncode}; see {log.name}")
                try:
                    with urllib.request.urlopen(url + "/v1/models", timeout=3) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("server did not become ready in 15 minutes")
                time.sleep(2)
            print("server ready", flush=True)
            if a.cancel_after_prefill:
                streaming = dict(request, stream=True, return_progress=True)
                req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(streaming).encode(),
                                             {"content-type": "application/json"})
                with urllib.request.urlopen(req, timeout=900) as response:
                    for line in response:
                        if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                            continue
                        event = json.loads(line[6:])
                        progress = event.get("prompt_progress") or {}
                        if progress.get("processed", 0) > 0:
                            print("disconnecting after prefill progress", progress, flush=True)
                            break
                    else:
                        raise AssertionError("stream ended without prefill progress")
            results = []
            for run in range(a.turns):
                started = time.monotonic()
                req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(request).encode(),
                                             {"content-type": "application/json"})
                with urllib.request.urlopen(req, timeout=900) as response:
                    result = json.load(response)
                message = result["choices"][0]["message"]
                assert message.get("content") or message.get("reasoning_content"), "empty completion"
                assert result["usage"]["completion_tokens"] > 0, "no generated tokens"
                results.append({"run": run + 1, "elapsed_s": time.monotonic() - started, "response": result})
                output.write_text(json.dumps(results, indent=2))
                print(json.dumps({k: v for k, v in results[-1].items() if k != "response"}),
                      json.dumps({k: result.get(k) for k in ("usage", "timings")}), flush=True)
                if a.followup:
                    request["messages"] += [
                        {"role": "assistant", "content": message.get("content") or "The reeve greets you."},
                        {"role": "user", "content": "I ask the reeve which road leads to the abbey. Continue the scene."}]
                if a.rewrite_history:
                    request["messages"][1]["content"] = f"World update for turn {run + 2}.\n" + request["messages"][1]["content"]
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


if __name__ == "__main__":
    main()
