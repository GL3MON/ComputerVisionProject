"""CPU-only regression checks for pooling, gradients, patch order, and attention."""

import unittest
from types import SimpleNamespace
import numpy as np
import torch
from medical_vlm.vision.features import image_embeddings, restore_patch_grid
from medical_vlm.vision.maps import summarize_map
from medical_vlm.vision.classifier import LinearClassifier, evaluate_classifier
from medical_vlm.vision.attribution import attribution_for_image


class VisionTests(unittest.TestCase):
    def test_packed_patches_are_restored_to_raster_order(self):
        actual = restore_patch_grid(torch.arange(16), (1, 4, 4), 2)
        expected = torch.tensor([[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]])
        self.assertTrue(torch.equal(actual, expected))
        with self.assertRaises(ValueError):
            restore_patch_grid(torch.arange(15), (1, 4, 4), 2)

    def test_pooling_supports_current_and_tuple_interfaces(self):
        embeddings = (torch.tensor([[1.0, 3.0], [3.0, 5.0]]), torch.tensor([[4.0, 6.0]]))
        for output in [SimpleNamespace(pooler_output=embeddings), (embeddings, [])]:
            model = SimpleNamespace(get_image_features=lambda **kwargs: output)
            actual = image_embeddings(model, {"pixel_values": None, "image_grid_thw": None})
            self.assertTrue(torch.equal(actual, torch.tensor([[2.0, 4.0], [4.0, 6.0]])))

    def test_mass_is_normalized_and_degenerate_maps_are_finite(self):
        metrics = summarize_map(np.array([[0.0, 0.0], [0.0, 10.0]]), 0.25)
        self.assertAlmostEqual(metrics["top_10_percent_mass"], 1.0)
        self.assertAlmostEqual(metrics["center_x"], 1.0)
        for value in summarize_map(np.zeros((1, 1))).values():
            self.assertTrue(np.isfinite(value))

    def test_classifier_learning_uses_shared_architecture(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)
        features = torch.tensor([[-1.0, -1.0], [1.0, 1.0]] * 4)
        labels = torch.tensor([0, 1] * 4)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(features, labels), batch_size=8
        )
        model = LinearClassifier(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
        before = evaluate_classifier(model, loader, "cpu")[0]["loss"]
        for _ in range(5):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(features), labels)
            loss.backward()
            optimizer.step()
        after = evaluate_classifier(model, loader, "cpu")[0]
        self.assertLess(after["loss"], before)
        self.assertEqual(after["accuracy"], 1.0)

    def test_gradient_attribution_has_expected_shape(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(24, 4)

            def get_image_features(self, pixel_values, **kwargs):
                return SimpleNamespace(pooler_output=(self.layer(pixel_values),))

        class Processor:
            merge_size = 2

            def __init__(self):
                self.image_processor = self

            def __call__(self, **kwargs):
                return {
                    "pixel_values": torch.ones(16, 24),
                    "image_grid_thw": torch.tensor([[1, 4, 4]]),
                }

        values, prediction, confidence = attribution_for_image(
            Model(), Processor(), LinearClassifier(4, 2), None
        )
        self.assertEqual(values.shape, (4, 4))
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue(0 <= confidence <= 1)

    def test_actual_visual_attention_is_captured_and_patch_is_restored(self):
        from transformers.models.qwen3_vl import modeling_qwen3_vl as implementation
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
        from medical_vlm.vision.attention import attention_for_image

        config = Qwen3VLVisionConfig(
            depth=2,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            out_hidden_size=16,
            patch_size=2,
            spatial_merge_size=2,
            temporal_patch_size=2,
            deepstack_visual_indexes=[0, 1],
            num_position_embeddings=16,
        )
        config._attn_implementation = "eager"
        visual = implementation.Qwen3VLVisionModel(config).eval()
        model = SimpleNamespace(model=SimpleNamespace(visual=visual), parameters=visual.parameters)

        class Processor:
            merge_size = 2

            def __init__(self):
                self.image_processor = self

            def __call__(self, **kwargs):
                return {
                    "pixel_values": torch.randn(16, 24),
                    "image_grid_thw": torch.tensor([[1, 4, 4]]),
                }

        original = implementation.eager_attention_forward
        values = attention_for_image(model, Processor(), None)
        self.assertEqual(values.shape, (4, 4))
        self.assertTrue(np.isfinite(values).all())
        self.assertAlmostEqual(float(values.sum()), 1.0, places=5)
        self.assertIs(implementation.eager_attention_forward, original)


if __name__ == "__main__":
    unittest.main()
