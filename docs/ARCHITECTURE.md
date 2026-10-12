# Architecture

Entry points parse arguments, load configuration, and call workflow modules. They do not contain model, scoring, or persistence implementations.

| Layer | Responsibility |
| --- | --- |
| `core` | Isolated configuration merges, project-root resolution, atomic JSON writes. |
| `data` | Dataset adapters and preprocessing; no training orchestration. |
| `evaluation.client` | An injected API client and immutable inference settings. Multiple model clients can coexist without changing global state. |
| `evaluation.scoring` | Stable PMC-VQA prompt and answer normalization for historical comparability. |
| `evaluation.benchmark` | Bounded request scheduling, resume validation, progress persistence, summaries. |
| `training` | Model/trainer orchestration; preprocessing and reward validation are independently testable. |
| `models` | Local/Hub model resolution, safe adapter merge, explicit upload, inspection, and serving. |
| `vision` | Shared pooled features and classifiers, raster patch restoration, normalized attribution metrics, and attention capture. |
| `reports` | Pure metric/statistical helpers and rendering, with loading and persistence handled by the report workflow. |

Paths resolve against the editable checkout. `MEDICAL_VLM_ROOT` explicitly overrides that root. Optional training/vision dependencies are loaded when their workflows are invoked. Package and CLI imports do not initiate training, create output directories, or contact servers.

Historical output files retain their paths and bytes. Root source files are archived before relocation. Generated reports use a separate destination by default and refuse overwrites. The original grading rules remain unchanged; corrected vision measurements are documented as new methodology rather than used to relabel historical results.
