"""One lazy command dispatcher for research workflows."""

import argparse
import importlib
import sys

COMMANDS = {
    "benchmark": "evaluation.benchmark",
    "train-sft": "training.sft",
    "train-grpo": "training.grpo",
    "rollouts": "evaluation.rollouts",
    "inspect-checkpoint": "models.inspect",
    "merge": "models.merge",
    "upload": "models.hub",
    "probe": "vision.probe",
    "attribution": "vision.attribution",
    "compare-attribution": "vision.compare",
    "attention": "vision.attention",
    "smoke": "evaluation.smoke",
    "report": "reports.cli",
    "serve": "models.serve",
}


def main():
    parser = argparse.ArgumentParser(
        prog="medical-vlm", description="Medical vision-language research workflows"
    )
    parser.add_argument("command", choices=sorted(COMMANDS))
    if len(sys.argv) > 1 and sys.argv[1] in COMMANDS:
        command = sys.argv[1]
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        importlib.import_module("medical_vlm." + COMMANDS[command]).main()
    else:
        parser.parse_args()
