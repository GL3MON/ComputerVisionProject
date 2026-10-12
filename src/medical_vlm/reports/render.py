"""Markdown rendering separated from loading, calculations, and persistence."""


def format_number(value):
    return "Not recorded" if value is None else f"{value:,.2f}"


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(str(value) for value in row) + " |" for row in rows],
        ]
    )


def render_combined(metrics):
    lines = [
        "# Combined PMC-VQA metrics",
        "",
        table(
            [
                "Model",
                "Questions",
                "Planned prompts",
                "Saved successful prompts",
                "Pass@1",
                "Best-of-k",
                "Failures",
            ],
            [
                [
                    m["model"],
                    m["question_examples"],
                    m["planned_inference_prompts"],
                    m["recorded_successful_prompts"],
                    f"{m['accuracy_pass1']:.2%}",
                    f"{m['accuracy_pass2']:.2%}",
                    m["failed_question_examples"],
                ]
                for m in metrics
            ],
        ),
        "",
        "A prompt is one retained generation attempt. Planned prompts include attempts not retained after failures. Actual network requests, retries, and discarded interrupted work are not reconstructed from result files.",
        "",
        table(
            ["Metric", *[m["model"] for m in metrics]],
            [
                [label, *[format_number(m[key]) for m in metrics]]
                for label, key in [
                    ("Input tokens", "prompt_tokens"),
                    ("Output tokens", "completion_tokens"),
                    ("Total tokens", "total_tokens"),
                    ("Mean output tokens per prompt", "mean_completion_tokens_per_recorded_prompt"),
                    ("Mean example latency (seconds)", "mean_example_latency_seconds"),
                    ("P95 example latency (seconds)", "p95_example_latency_seconds"),
                ]
            ],
        ),
        "",
        "Historical serving and processor configurations are not fully recorded. Input-token differences cannot be attributed solely to training, and overlapping request latencies are not elapsed wall time.",
        "",
        "These tables are regenerated from saved predictions without rerunning inference.",
    ]
    return "\n".join(lines) + "\n"


def render_analysis(metrics, comparisons):
    current, previous = metrics[-1], metrics[-2]
    first = (current["accuracy_pass1"] - previous["accuracy_pass1"]) * 100
    best = (current["accuracy_pass2"] - previous["accuracy_pass2"]) * 100
    recovered = current["correct_pass2_count"] - current["correct_pass1_count"]
    lines = [
        "# Benchmark analysis",
        "",
        f"{current['model']} versus {previous['model']}: {first:+.2f} percentage points at pass@1 and {best:+.2f} percentage points in best-of-k coverage.",
        "",
        "## Paired outcomes",
        "",
        table(
            [
                "Metric",
                "Current-only correct",
                "Previous-only correct",
                "Net change",
                "Exact McNemar p",
            ],
            [
                [
                    name,
                    p["current_only_correct"],
                    p["previous_only_correct"],
                    p["net_current_correct_change"],
                    f"{p['exact_mcnemar_pvalue']:.4f}",
                ]
                for name, p in comparisons.items()
            ],
        ),
        "",
        "The test is an exploratory question-level matched-outcome comparison, without image-cluster adjustment or repeated sampling/training seeds. Failure to reject a difference does not establish equivalence. [Exact-test method](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html).",
        "",
        "## Prompt budget and recovery",
        "",
        f"The current run evaluates {current['question_examples']:,} questions and plans {current['planned_inference_prompts']:,} prompts, retaining {current['recorded_successful_prompts']:,} successful attempts. Additional attempts cover {recovered} first-attempt misses.",
        "",
        "Best-of-k is oracle coverage determined using the correct answer. It is not the accuracy of an automatically selected final response; a selection or verification policy needs its own evaluation.",
        "",
        "## Tokens and request coverage",
        "",
        f"The current run records {current['prompt_tokens']:,} input tokens, {current['completion_tokens']:,} output tokens, and {current['failed_question_examples']} failed examples. Input and output totals are separate from prompt counts.",
        "",
        "Matched questions do not establish identical image preprocessing or tokenization. Historical server settings and wall times are incomplete. Control processor settings, image resolution, hardware, sampling, and concurrency before attributing efficiency or accuracy differences to training.",
        "",
        "## Next checks",
        "",
        "Repeat with recorded seeds, inspect paired gains and regressions, retain finish reasons and actual request counts, and compare the selected final answer when multiple prompts are used.",
    ]
    return "\n".join(lines) + "\n"
