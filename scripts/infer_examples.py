#!/usr/bin/env python3
"""Run the prepared SemBridge demo examples with a single model load."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List examples without loading a model")
    parser.add_argument("--task", choices=("all", "tts", "svs", "unified"), default="all")
    parser.add_argument("--example", nargs="+", metavar="ID", help="Run selected IDs from --list")
    parser.add_argument("--model", default=os.environ.get("MODEL", "ASLP-lab/SemBridge"))
    parser.add_argument("--output-dir", type=Path, default=os.environ.get("OUTPUT_DIR", "outputs/examples"))
    parser.add_argument("--decoder", choices=("auto", "native", "bigvgan"),
                        default=os.environ.get("DECODER", "auto"))
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"),
                        default=os.environ.get("DTYPE", "bfloat16"))
    parser.add_argument("--revision")
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-patches", type=int, default=os.environ.get("MAX_PATCHES"))
    parser.add_argument("--min-patches", type=int, default=os.environ.get("MIN_PATCHES"))
    parser.add_argument("--flow-steps", type=int, default=os.environ.get("FLOW_STEPS"))
    parser.add_argument("--cfg-value", type=float, default=os.environ.get("CFG_VALUE"))
    parser.add_argument("--temperature", type=float, default=os.environ.get("TEMPERATURE"))
    parser.add_argument("--seed", type=int, default=os.environ.get("SEED"))
    args = parser.parse_args(argv)
    entries = json.loads((EXAMPLES / "index.json").read_text(encoding="utf-8"))["examples"]
    if args.example:
        unknown = set(args.example) - {entry["id"] for entry in entries}
        if unknown:
            parser.error(f"unknown example IDs: {', '.join(sorted(unknown))}; use --list")
        entries = [entry for entry in entries if entry["id"] in args.example]
        if args.task != "all" and any(entry["task"] != args.task for entry in entries):
            parser.error("selected example IDs do not all match --task")
    if args.task != "all":
        entries = [entry for entry in entries if entry["task"] == args.task]
    if args.list:
        print(f"{'ID':<30} {'Task':<8} {'Language':<10} Request")
        for entry in entries:
            print(f"{entry['id']:<30} {entry['task']:<8} {entry['language']:<10} {entry['request']}")
        return 0

    # Resolve every request before loading weights, including when run outside the repo.
    sys.path.insert(0, str(ROOT))
    from sembridge import SemBridge
    from sembridge.requests import load_request

    overrides = {
        "max_new_patches": args.max_patches, "min_new_patches": args.min_patches,
        "flow_steps": args.flow_steps, "cfg_value": args.cfg_value,
        "temperature": args.temperature, "seed": args.seed,
    }
    requests = []
    for entry in entries:
        effective = dict(overrides)
        task_maximum = os.environ.get(f"{entry['task'].upper()}_MAX_PATCHES")
        try:
            if args.max_patches is None and task_maximum is not None:
                effective["max_new_patches"] = int(task_maximum)
            request = load_request(EXAMPLES / entry["request"], task=entry["task"],
                                   generation_overrides=effective)
        except (OSError, ValueError) as exc:
            parser.error(f"{entry['id']}: {exc}")
        requests.append((entry, request))

    print(f"Loading {args.model} on {args.device} ({args.dtype})", flush=True)
    model = SemBridge.from_pretrained(
        args.model, device=args.device, dtype=args.dtype, revision=args.revision,
        cache_dir=args.cache_dir, local_files_only=args.local_files_only,
    )
    for index, (entry, request) in enumerate(requests, start=1):
        destination = args.output_dir.expanduser() / f"{entry['id']}.wav"
        print(f"[{index}/{len(requests)}] {entry['id']} -> {destination}", flush=True)
        model.generate_to_file(destination, decoder=args.decoder, **request)
    print(f"Saved {len(requests)} audio files to {args.output_dir.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
