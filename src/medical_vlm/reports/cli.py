"""Generate combined metrics and paired analysis from configurable saved runs."""

import argparse
import csv
from medical_vlm.core.config import load_config
from medical_vlm.core.paths import project_root, resolve_path
from medical_vlm.core.io import read_json, write_json
from medical_vlm.reports.metrics import collect
from medical_vlm.reports.statistics import paired
from medical_vlm.reports.render import render_combined, render_analysis


def build_reports(config, output, overwrite=False):
    specs = config["runs"]
    if len(specs) < 2:
        raise ValueError("At least two runs are required")
    output = resolve_path(output)
    names = [
        "COMBINED_REPORT.md",
        "ANALYSIS_REPORT.md",
        "combined_metrics.json",
        "combined_metrics.csv",
        "analysis_metrics.json",
    ]
    if not overwrite and any((output / name).exists() for name in names):
        raise FileExistsError(
            "Report artifacts already exist; choose a new --output-dir or explicitly use --overwrite"
        )
    rows = [read_json(resolve_path(spec["results"])) for spec in specs]
    reference = {r["index"]: r for r in rows[0]}
    for items in rows[1:]:
        if {r["index"] for r in items} != set(reference):
            raise ValueError("Result indices differ between runs")
        if not all(
            all(
                r[key] == reference[r["index"]][key]
                for key in ["question", "options", "ground_truth"]
            )
            for r in items
        ):
            raise ValueError("Questions, options, or answers do not align")
    metrics = [
        collect(
            spec["name"],
            spec["results"],
            items,
            resolve_path(spec["summary"]) if spec.get("summary") else None,
        )
        for spec, items in zip(specs, rows)
    ]
    common = set.intersection(*[{r["index"] for r in items if "error" not in r} for items in rows])
    for metric, items in zip(metrics, rows):
        shared = [r for r in items if r["index"] in common]
        metric.update(
            common_successful_examples=len(shared),
            accuracy_pass1_common_subset=sum(r["correct_pass1"] for r in shared) / len(shared)
            if shared
            else None,
            accuracy_pass2_common_subset=sum(r["correct_passk"] for r in shared) / len(shared)
            if shared
            else None,
        )
    left, right = [{r["index"]: r for r in items} for items in rows[-2:]]
    comparisons = {
        name: paired(left, right, key, sorted(reference))
        for name, key in [("pass@1", "correct_pass1"), ("best-of-k", "correct_passk")]
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "combined_metrics.json",
        {"models": metrics, "aligned_questions_options_answers": True},
    )
    write_json(
        output / "analysis_metrics.json",
        {
            "comparisons": comparisons,
            "previous_model": specs[-2]["name"],
            "current_model": specs[-1]["name"],
        },
    )
    with (output / "combined_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    (output / "COMBINED_REPORT.md").write_text(render_combined(metrics))
    (output / "ANALYSIS_REPORT.md").write_text(render_analysis(metrics, comparisons))
    return metrics, comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(project_root() / "configs/reports/pmc_vqa.yaml"))
    parser.add_argument("--output-dir", default="outputs/benchmarks/pmc_vqa/reports")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    metrics, _ = build_reports(load_config(args.config), args.output_dir, args.overwrite)
    for metric in metrics:
        print(
            f"{metric['model']}: pass@1={metric['accuracy_pass1']:.2%}; best-of-k={metric['accuracy_pass2']:.2%}"
        )


if __name__ == "__main__":
    main()
