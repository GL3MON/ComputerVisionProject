"""Offline regression checks for scoring, configuration, rewards, and reporting."""

import json
import io
import sys
from contextlib import redirect_stdout, redirect_stderr
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from PIL import Image
from medical_vlm.core.config import deep_merge, load_config
from medical_vlm.core.io import write_json, read_json
from medical_vlm.core.paths import project_root, resolve_path
from medical_vlm.data.reasoning import preprocess_example
from medical_vlm.evaluation.client import ChatInference, InferenceSettings
from medical_vlm.evaluation.benchmark import evaluate_sample, summarize
from medical_vlm.evaluation.scoring import normalize_answer, extract_answer
from medical_vlm.training.rewards import VerifierReward, parse_verifier_output
from medical_vlm.reports.metrics import collect
from medical_vlm.reports.statistics import paired, exact_mcnemar
from medical_vlm.reports.render import render_combined


class FakeService:
    def __init__(self, answers):
        self.answers = answers

    def generate_n(self, sample, k):
        return [
            {
                "normalized": answer,
                "predicted": answer,
                "reasoning": None,
                "response": f"<answer>{answer}</answer>",
                "latency": 1.0,
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "finish_reason": "stop",
            }
            for answer in self.answers[:k]
        ]


def sample(answer="B"):
    return {"question": "Choose an option", "options": ["A: one", "B: two"], "answer": answer}


class CoreTests(unittest.TestCase):
    def test_config_defaults_do_not_leak_mutations(self):
        base = {"nested": {"items": [1]}, "value": 3}
        merged = deep_merge(base, {"value": 4})
        merged["nested"]["items"].append(2)
        self.assertEqual(base, {"nested": {"items": [1]}, "value": 3})

    def test_invalid_yaml_mapping_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("- not-a-mapping\n")
            with self.assertRaises(ValueError):
                load_config(path)

    def test_atomic_json_roundtrip_and_no_temp_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested/result.json"
            write_json(path, {"value": "µ"})
            self.assertEqual(read_json(path), {"value": "µ"})
            self.assertFalse(path.with_name(path.name + ".tmp").exists())

    def test_project_paths_are_cwd_independent(self):
        self.assertTrue((project_root() / "pyproject.toml").exists())
        self.assertEqual(resolve_path("outputs"), project_root() / "outputs")

    def test_original_scoring_rules(self):
        self.assertEqual(normalize_answer(extract_answer("reason <answer>B: MRI</answer>")), "B")
        self.assertEqual(normalize_answer("D) finding"), "D")
        self.assertEqual(normalize_answer("unknown"), "unknown")

    def test_two_attempt_recovery_and_token_counts(self):
        result = evaluate_sample((0, sample()), FakeService(["A", "B"]), 2)
        self.assertFalse(result["correct_pass1"])
        self.assertTrue(result["correct_passk"])
        self.assertEqual(result["prompt_tokens"], 20)
        summary = summarize([result], {}, 2.0, True)
        self.assertEqual(summary["accuracy_passk"], 1.0)
        self.assertEqual(summary["truncated_attempts"], 0)

    def test_failure_in_denominator_and_empty_summary(self):
        service = Mock()
        service.generate_n.side_effect = RuntimeError("unavailable")
        result = evaluate_sample((0, sample()), service, 2)
        summary = summarize([result], {}, 1.0, True)
        self.assertEqual(summary["failed_examples"], 1)
        self.assertEqual(summary["accuracy_pass1"], 0.0)
        self.assertIsNone(summary["average_example_latency_seconds"])
        self.assertIsNone(summarize([], {}, 1.0, False)["accuracy_pass1"])

    def test_inference_clients_do_not_share_model_state(self):
        fake = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=Mock(
                        return_value=SimpleNamespace(
                            choices=[
                                SimpleNamespace(
                                    message=SimpleNamespace(
                                        content="<answer>B</answer>", reasoning=None
                                    ),
                                    finish_reason="stop",
                                )
                            ],
                            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
                        )
                    )
                )
            )
        )
        one = ChatInference(fake, InferenceSettings("model-one"))
        two = ChatInference(fake, InferenceSettings("model-two"))
        row = {**sample(), "image": Image.new("RGB", (2, 2))}
        one.generate(row)
        two.generate(row)
        self.assertEqual(
            [call.kwargs["model"] for call in fake.chat.completions.create.call_args_list],
            ["model-one", "model-two"],
        )

    def test_assistant_only_label_masking(self):
        class Tokenizer:
            def __call__(self, text, **kwargs):
                ids = [1, 2] if text == "prompt" else [1, 2, 3, 4]
                return {"input_ids": ids, "attention_mask": [1] * len(ids)}

        class Processor:
            tokenizer = Tokenizer()

            def apply_chat_template(self, messages, **kwargs):
                return "prompt" if kwargs["add_generation_prompt"] else "full"

        encoded = preprocess_example(
            {"Question": "q", "Complex_CoT": "reason", "Response": "answer"}, Processor(), 4096
        )
        self.assertEqual(encoded["labels"], [-100, -100, 3, 4])

    def test_checklist_is_authoritative(self):
        good = json.dumps({"correct": False, "items": [{"required": "component", "present": True}]})
        bad = json.dumps({"correct": True, "items": [{"required": "component", "present": False}]})
        self.assertTrue(parse_verifier_output(good)[0])
        self.assertFalse(parse_verifier_output(bad)[0])
        self.assertFalse(parse_verifier_output("not JSON")[0])

    def test_reward_skips_unreasoned_answers(self):
        response = Mock()
        response.json.return_value = {
            "choices": [
                {"index": 0, "text": json.dumps({"items": [{"required": "B", "present": True}]})}
            ]
        }
        with patch("medical_vlm.training.rewards.requests.post", return_value=response) as request:
            reward = VerifierReward("http://fake", "verifier")
            values = reward.verifier_reward(["q", "q"], ["<think>reason</think>B", "B"], ["B", "B"])
            self.assertEqual(values, [1.0, -1.0])
            self.assertEqual(len(request.call_args.kwargs["json"]["prompt"]), 1)

    def test_metrics_and_paired_outcomes(self):
        left = [
            evaluate_sample((0, sample()), FakeService(["A", "B"])),
            evaluate_sample((1, sample("A")), FakeService(["A", "A"])),
        ]
        right = [
            evaluate_sample((0, sample()), FakeService(["B", "B"])),
            evaluate_sample((1, sample("A")), FakeService(["B", "B"])),
        ]
        metrics = collect("mock", "mock.json", left)
        self.assertEqual(metrics["recorded_successful_prompts"], 4)
        self.assertEqual(metrics["correct_pass2_count"], 2)
        result = paired(
            {r["index"]: r for r in left}, {r["index"]: r for r in right}, "correct_pass1", [0, 1]
        )
        self.assertEqual(result["net_current_correct_change"], 0)
        self.assertEqual(result["exact_mcnemar_pvalue"], 1.0)
        self.assertAlmostEqual(exact_mcnemar(297, 280), 0.5053887589167069)

    def test_benchmark_cli_saves_resumes_and_rejects_overwrites(self):
        from medical_vlm.evaluation import benchmark

        class Dataset(list):
            _fingerprint = "fixture-v1"

        dataset = Dataset(
            [{**sample(), "id": str(i), "image": Image.new("RGB", (2, 2))} for i in range(2)]
        )
        create = Mock(
            return_value=SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="<answer>B</answer>", reasoning=None),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            )
        )
        client = SimpleNamespace(
            models=SimpleNamespace(list=lambda: SimpleNamespace(data=[SimpleNamespace(id="mock")])),
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.json"
            argv = ["benchmark", "--output", str(output), "--model", "mock", "--workers", "2"]
            with (
                patch.dict(
                    sys.modules,
                    {
                        "datasets": SimpleNamespace(
                            load_dataset=lambda *args, **kwargs: {"test": dataset}
                        )
                    },
                ),
                patch.object(benchmark, "create_client", return_value=client),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                with patch.object(sys, "argv", argv):
                    benchmark.main()
                summary = read_json(output.with_name("results_summary.json"))
                self.assertTrue(summary["complete"])
                self.assertEqual(summary["evaluated_examples"], 2)
                self.assertEqual(create.call_count, 4)
                saved = output.read_bytes()
                with patch.object(sys, "argv", argv + ["--resume"]):
                    benchmark.main()
                self.assertEqual(create.call_count, 4)
                self.assertEqual(output.read_bytes(), saved)
                with patch.object(sys, "argv", argv), self.assertRaises(SystemExit) as raised:
                    benchmark.main()
                self.assertEqual(raised.exception.code, 2)
                self.assertEqual(output.read_bytes(), saved)
                with (
                    patch.object(sys, "argv", argv + ["--resume", "--max-tokens", "8"]),
                    self.assertRaises(SystemExit),
                ):
                    benchmark.main()
                self.assertEqual(output.read_bytes(), saved)

    def test_reports_handle_all_failed_runs(self):
        service = Mock()
        service.generate_n.side_effect = RuntimeError("unavailable")
        rows = [evaluate_sample((i, sample()), service) for i in range(2)]
        metrics = collect("failed", "mock.json", rows)
        self.assertIn("Not recorded", render_combined([metrics]))


if __name__ == "__main__":
    unittest.main()
