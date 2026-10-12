"""Compatibility entry point; canonical report code lives in medical_vlm.reports."""
from pathlib import Path
import sys

for parent in Path(__file__).resolve().parents:
    if (parent / "src/medical_vlm").is_dir():
        sys.path.insert(0,str(parent / "src"))
        break
from medical_vlm.reports.cli import main

if __name__ == "__main__":
    main()
