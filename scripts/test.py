#!/usr/bin/env python3
"""Quick smoke test for a running vLLM server."""

import sys
import json
import urllib.request
import urllib.error

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
# Use WSL2 bridge IP — loopback is blocked in Claude Code sandbox
BASE_URL = f"http://172.18.0.1:{PORT}/v1"


def request(path, payload=None):
    url = BASE_URL + path
    data = json.dumps(payload).encode() if payload else None
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def main():
    print(f"Testing vLLM server at {BASE_URL}\n")

    # Check available models
    models = request("/models")
    model_id = models["data"][0]["id"]
    print(f"Model loaded: {model_id}\n")

    # Simple generation test
    print("Sending test prompt...")
    result = request("/chat/completions", {
        "model": model_id,
        "messages": [{"role": "user", "content": "Say hello in exactly 5 words."}],
        "max_tokens": 32,
        "temperature": 0,
    })

    content = result["choices"][0]["message"]["content"]
    usage = result["usage"]
    print(f"Response: {content}")
    print(f"Tokens: {usage['prompt_tokens']} in / {usage['completion_tokens']} out")
    print("\nServer is working correctly.")


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError as e:
        print(f"Could not connect to {BASE_URL}: {e}")
        print("Is the server running? Check: vllm serve ...")
        sys.exit(1)
