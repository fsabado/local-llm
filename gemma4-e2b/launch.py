#!/usr/bin/env python3
"""
Gemma 4 E2B llama-server launcher.

Named flags, presets, and idempotent — only one server per port at a time.

Usage:
  python3 launch.py                    # default preset (20p)
  python3 launch.py --preset 16p
  python3 launch.py --preset fast
  python3 launch.py --parallel 12 --ctx-per-slot 65536
  python3 launch.py --restart          # kill existing, relaunch
  python3 launch.py --list-presets
"""

import argparse
import os
import signal
import socket
import subprocess
import sys
import time

LLAMA_BIN = os.path.expanduser("~/src/llama.cpp-upstream/build-cuda/bin")
DEFAULT_MODEL = os.path.expanduser(
    "~/models/gemma-4-e2b-it/gemma-4-E2B-it-Q4_K_M.gguf"
)
DEFAULT_PORT = 10222

# iSWA KV: 3 full-attn layers × 768 MiB/slot at 131K + 2.88 GiB model base
PRESETS = {
    "20p":    dict(parallel=20, ctx_per_slot=131072, vram_gb=19.3, desc="20 slots × 131K — agentic sweet spot"),
    "16p":    dict(parallel=16, ctx_per_slot=131072, vram_gb=17.9, desc="16 slots × 131K"),
    "40p":    dict(parallel=40, ctx_per_slot=131072, vram_gb=24.0, desc="40 slots × 131K — iSWA ceiling"),
    "4p":     dict(parallel=4,  ctx_per_slot=131072, vram_gb=8.6,  desc="4 slots × 131K — max context quality"),
    "single": dict(parallel=1,  ctx_per_slot=2097152, vram_gb=21.7, desc="1 slot × 2M — max single-user context"),
    "fast":   dict(parallel=64, ctx_per_slot=2048,   vram_gb=7.0,  desc="64 slots × 2K — max concurrency, short ctx"),
}
DEFAULT_PRESET = "20p"


def find_pids_on_port(port: int) -> list[int]:
    result = subprocess.run(
        ["lsof", "-ti", f"tcp:{port}"],
        capture_output=True, text=True
    )
    if result.returncode != 0 or not result.stdout.strip():
        return []
    return [int(p) for p in result.stdout.strip().split()]


def port_is_open(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def kill_server(port: int) -> None:
    pids = find_pids_on_port(port)
    if not pids:
        print(f"  No process found on port {port}.")
        return
    for pid in pids:
        print(f"  Killing PID {pid} on port {port}...")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    # Wait for port to free up
    for _ in range(20):
        time.sleep(0.5)
        if not port_is_open(port):
            print("  Server stopped.")
            return
    print("  SIGTERM timed out, sending SIGKILL...")
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def launch(model: str, port: int, parallel: int, ctx_per_slot: int) -> None:
    total_ctx = parallel * ctx_per_slot
    llama_bin = os.path.expanduser(LLAMA_BIN)
    server = os.path.join(llama_bin, "llama-server")

    print(f"Model:    {os.path.basename(model)}")
    print(f"Port:     {port}")
    print(f"Parallel: {parallel} slots × {ctx_per_slot:,} tokens = {total_ctx:,} total ctx")
    print()

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = llama_bin + ":" + env.get("LD_LIBRARY_PATH", "")

    os.execve(server, [
        server,
        "--model", model,
        "--port", str(port),
        "--host", "0.0.0.0",
        "-ngl", "99",
        "--ctx-size", str(total_ctx),
        "--parallel", str(parallel),
        "--cont-batching",
        "--flash-attn", "on",
        "--no-warmup",
        "--log-disable",
    ], env)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gemma 4 E2B llama-server launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--preset", choices=PRESETS.keys(), default=DEFAULT_PRESET,
                        help=f"Named config preset (default: {DEFAULT_PRESET})")
    parser.add_argument("--parallel", type=int, help="Number of parallel slots (overrides preset)")
    parser.add_argument("--ctx-per-slot", type=int, dest="ctx_per_slot",
                        help="Context tokens per slot (overrides preset)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--restart", action="store_true",
                        help="Kill existing server on port before launching")
    parser.add_argument("--list-presets", action="store_true",
                        help="Print available presets and exit")
    args = parser.parse_args()

    if args.list_presets:
        print(f"{'PRESET':<10} {'PARALLEL':>8} {'CTX/SLOT':>10} {'VRAM':>8}  DESCRIPTION")
        print("-" * 70)
        for name, p in PRESETS.items():
            marker = " *" if name == DEFAULT_PRESET else ""
            print(f"{name:<10} {p['parallel']:>8} {p['ctx_per_slot']:>10,} {p['vram_gb']:>7.1f}G  {p['desc']}{marker}")
        return

    preset = PRESETS[args.preset]
    parallel = args.parallel or preset["parallel"]
    ctx_per_slot = args.ctx_per_slot or preset["ctx_per_slot"]
    model = os.path.expanduser(args.model)
    port = args.port

    if port_is_open(port):
        if args.restart:
            print(f"Server already running on port {port} — restarting...")
            kill_server(port)
        else:
            print(f"Server already running on port {port}. Use --restart to replace it.")
            sys.exit(0)

    launch(model, port, parallel, ctx_per_slot)


if __name__ == "__main__":
    main()
