"""Command-line inference for speech, singing, and mixed generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .requests import load_request, resolve_generation_options


def infer_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("tts", "svs", "unified"), required=True)
    parser.add_argument("--model", default="ASLP-lab/SemBridge",
                        help="Local model directory or Hugging Face model ID")
    parser.add_argument("--input", help="Request JSON file")
    parser.add_argument("--output", default="outputs/generated.wav")
    parser.add_argument("--decoder", choices=("auto", "native", "bigvgan"), default="auto")
    parser.add_argument("--revision", help="Pin a Hugging Face model revision")
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true",
                        help="Load the model and check compatibility without generating audio")
    parser.add_argument("--max-patches", type=int)
    parser.add_argument("--min-patches", type=int)
    parser.add_argument("--flow-steps", type=int)
    parser.add_argument("--cfg-value", type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    args = parser.parse_args(argv)

    generation_overrides = {
        "max_new_patches": args.max_patches, "min_new_patches": args.min_patches,
        "flow_steps": args.flow_steps, "cfg_value": args.cfg_value,
        "temperature": args.temperature, "seed": args.seed,
    }
    load_options = {
        "device": args.device, "dtype": args.dtype, "revision": args.revision,
        "cache_dir": args.cache_dir, "local_files_only": args.local_files_only,
    }
    from .pipeline import SemBridge

    if args.validate_only:
        try:
            resolve_generation_options(overrides=generation_overrides)
        except ValueError as exc:
            parser.error(str(exc))
        pipeline = SemBridge.from_pretrained(args.model, **load_options)
        print(json.dumps(pipeline.validation_summary(task=args.task, decoder=args.decoder),
                         sort_keys=True))
        return 0
    if not args.input:
        parser.error("--input is required for generation")
    if Path(args.output).suffix.lower() != ".wav":
        parser.error("--output must end in .wav")
    try:
        request = load_request(args.input, task=args.task,
                               generation_overrides=generation_overrides)
    except (OSError, ValueError) as exc:
        parser.error(f"cannot read request: {exc}")

    pipeline = SemBridge.from_pretrained(args.model, **load_options)
    output = pipeline.generate_to_file(
        args.output,
        decoder=args.decoder,
        **request,
    )
    print(output)
    return 0
