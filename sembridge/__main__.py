"""Run inference with ``python -m sembridge infer``."""

import argparse
import sys

from .cli import infer_main


def main() -> int:
    # Forward help to the inference parser instead of consuming it here.
    if len(sys.argv) > 1 and sys.argv[1] == "infer":
        return infer_main(sys.argv[2:])
    parser = argparse.ArgumentParser(prog="python -m sembridge")
    parser.add_argument("command", choices=("infer",))
    parser.parse_args()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
