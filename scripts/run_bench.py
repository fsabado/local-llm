#!/usr/bin/env python3
"""Start llama-server for a given GGUF, run bench, kill server."""

import sys, os, time, subprocess, urllib.request, json

LLAMA_BIN = "/home/fsabado/src/llama-turboquant-animehacker/build-cuda/bin"
PORT = 10111
BASE = f"http://172.18.0.1:{PORT}/v1"
GGUF_DIR = "/home/fsabado/src/local-llm/qwen35-27b-gguf"
CHAT_TEMPLATE = f"{GGUF_DIR}/chat_template.jinja"

# 65536 = 64K per slot (p=4). Matches Q4_K_M p=4 per-slot config for fair comparison.
CTX_SIZE = "65536"

def get_vram_used_mib():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True
        ).strip()
        return int(out)
    except Exception:
        return -1

def wait_for_server(timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://172.18.0.1:{PORT}/health", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False

def kill_server(proc, log_file=None):
    if proc and proc.poll() is None:
        proc.kill()
        proc.wait(timeout=15)
    if log_file:
        log_file.close()
    # Wait for VRAM to be released (up to 20s)
    deadline = time.time() + 20
    while time.time() < deadline:
        used = get_vram_used_mib()
        if used < 3000:  # baseline ~1500 MiB
            break
        time.sleep(2)
    time.sleep(2)  # extra settle

def run_bench(model_path, label):
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = LLAMA_BIN + ":" + env.get("LD_LIBRARY_PATH", "")

    log_path = f"/home/fsabado/.claude/tmp/llama_server_{os.path.basename(model_path)}.log"

    cmd = [
        f"{LLAMA_BIN}/llama-server",
        "--model", model_path,
        "--port", str(PORT),
        "--host", "0.0.0.0",
        "--ctx-size", CTX_SIZE,
        "--n-gpu-layers", "99",
        "--parallel", "4",
        "--cont-batching",
        "--flash-attn", "on",
        "--cache-type-k", "tq3_0",
        "--cache-type-v", "tq3_0",
        "--jinja",
        "--chat-template-file", CHAT_TEMPLATE,
        "--reasoning-format", "deepseek",
    ]

    vram_before = get_vram_used_mib()
    print(f"\nStarting server: {os.path.basename(model_path)}")
    print(f"Context: {CTX_SIZE} | VRAM before: {vram_before} MiB | Log: {log_path}")

    log_file = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=log_file)

    print("Waiting for server ready...")
    if not wait_for_server(timeout=180):
        print("ERROR: Server did not start in time. Checking log...")
        log_file.flush()
        with open(log_path) as f:
            tail = f.readlines()[-20:]
        print("".join(tail))
        kill_server(proc, log_file)
        return None

    vram_after = get_vram_used_mib()
    print(f"Server ready. VRAM now: {vram_after} MiB (+{vram_after - vram_before} MiB)\n")

    # Import bench module, override BASE/MODEL and patch post() for longer timeout
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    saved_argv = sys.argv[:]
    sys.argv = [sys.argv[0]]
    import importlib
    import bench as _bench
    importlib.reload(_bench)  # reload so module-level PORT/BASE/MODEL re-execute cleanly
    _bench.BASE = BASE
    _bench.MODEL = "qwen35-27b"
    sys.argv = saved_argv

    def _post_long(path, payload=None):
        url = _bench.BASE + path
        data = json.dumps(payload).encode() if payload else None
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())
    _bench.post = _post_long

    results = _bench.run_full_benchmark(label=label)

    kill_server(proc, log_file)
    print(f"Server stopped. VRAM after cleanup: {get_vram_used_mib()} MiB")
    return results

if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled.i1-Q3_K_M.gguf"
    model_path = os.path.join(GGUF_DIR, model_name)
    label = os.path.basename(model_path).replace(".gguf", "")
    run_bench(model_path, label)
