# Checkpoint 2100 vision encoder experiment

Completed on MIG 2 (`MIG-be0d4dc8-2244-5ed0-89ec-b674eacb6a9b`).

Checkpoint: `/home/harishr/MedGRPO/qwen3-vl-medical-grpo-vision-lora/checkpoint-2100`
Base: `GL3MON/qwen3-vl-2b-medical-pmcvqa-sft-merged`

The checkpoint's vision and language LoRA adapters were loaded and merged. The exported vision tower includes the vision LoRA updates. A reconstructed first-block QKV weight exactly matches the export; it differs from the base weight. Features and attribution maps were checked for finite values, expected sizes, and successful stage exits.

## Frozen encoder and linear classifier

PneumoniaMNIST at 224 pixels with the checkpoint processor; the processor applies its configured resize. Extracted 2,048-dimensional embeddings by averaging merged vision tokens. All encoder parameters stayed frozen during classifier training. The encoder weights come from the supplied GRPO checkpoint.

Train/validation/test: 4,708 / 524 / 624 images. Seed 3407; 20 epochs; AdamW, learning rate 0.001, weight decay 0.0001; classifier batch size 256. Best classifier selected by validation accuracy.

| Metric | Result |
|---|---:|
| Best epoch | 19 |
| Validation accuracy | 95.42% |
| Test accuracy | 89.74% |
| Test weighted F1 | 0.8941 |
| Test AUROC | 0.9791 |
| Normal recall | 74.79% |
| Pneumonia recall | 98.72% |

Confusion matrix (true rows, predicted columns; normal=0, pneumonia=1): `[[175, 59], [5, 385]]`. Errors are mainly normal images classified as pneumonia (59), versus five missed pneumonia images.

## Attribution

25 seeded validation images; absolute gradient times normalized input for the predicted class logit. Packed patches are restored to spatial raster order. Raw maps, three-panel figures, and per-image CSV metrics are saved. These maps explain the trained linear classifier's output; they do not measure localization accuracy or explain generated medical answers.

| Metric | Mean |
|---|---:|
| Accuracy on sampled images | 84.00% |
| Predicted class confidence | 0.8878 |
| Normalized attribution entropy | 0.8910 |
| Fraction of mass in top 10% of patches | 41.21% |
| Attribution center (x, y) | (0.507, 0.607) |

High normalized entropy indicates spatially spread attribution, while the most active tenth of patches carries about 41% of total mass. No matched base-model experiment was run, so this run alone does not establish improvement from GRPO. Earlier scripts' top-10% metric used an unnormalized sum, and their patch ordering differed; those metrics and visual positions are not directly comparable to this corrected analysis.

## Artifacts

- `encoder/vision_encoder.safetensors`: merged visual tower; `vision_config.json` and `processor/` accompany it.
- `encoder/verification.json`: vision LoRA merge verification.
- `classifier/embeddings/`: all split embeddings and labels.
- `classifier/linear_classifier_best.pt`, `linear_classifier_final.pt`: classifier weights and metadata.
- `classifier/metrics.json`, `test_predictions.npz`: training history, classification metrics, and test predictions.
- `attribution/`: 25 PNG maps, 25 raw NumPy arrays, CSV metrics, and JSON summary.
- `manifest.json`, `status.json`, `validation.json`, and stage logs: provenance and checks.

To reproduce, run `train_probe.py` followed by `analyze_attribution.py` with the med environment Python and the MIG UUID above in `CUDA_VISIBLE_DEVICES`. These scripts use absolute output paths and overwrite their own experiment artifacts when rerun.
