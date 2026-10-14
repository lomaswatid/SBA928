# SBA 928 - Market Research AI

Packaged Python project for CAX215. It studies prompt engineering and genuine FLAN-T5-small fine-tuning on UCI shopping-session statistics. **Recorded results do not demonstrate improved answer quality after fine-tuning.**

Start with [FINAL_REPORT.md](FINAL_REPORT.md). The report covers all six assessment requirements and the verified limitations of custom inference.

## Evidence and project layout

```text
src/sba_928/                    Package and six assignment steps
tests/                         Existing 207-test suite
data/online_shoppers_intention.csv  Public UCI source snapshot
data/processed/                Frozen 18 train / 3 validation / 3 test examples
data/evaluation/dev_pilot_v1/   Frozen 16-case synthetic development pilot
prompts/                       Twelve-question catalog and four prompt styles
submission/evidence/           Small public copies of recorded experiment outputs
submission/evidence_manifest.json  Original/public hashes and export mapping
docs/evaluation_framework.md   Existing evaluation design and limitations
FINAL_REPORT.md                Academic report draft
pyproject.toml, uv.lock        Package definition and locked dependencies
```

`artifacts/`, `models/`, and `results/` are local experiment archives, excluded from Git. Public evidence copies preserve raw responses and scores while redacting local filesystem paths. Course-message copies, caches, credentials, model weights and optimizer states are also excluded. Do not confuse legacy `prompts/train.jsonl` or older one-epoch results with the current manifest-backed dataset.

## Dataset and attribution

The source is [UCI dataset 468](https://doi.org/10.24432/C5F88Q), credited to Sakar and Kastro (2018), under CC BY 4.0. [data/README.md](data/README.md) explains attribution and the frozen splits. Local analysis contains 12,330 sessions, 17 features and the Revenue outcome; 1,908 sessions purchased. No verified demographic or competitor-company fields are provided.

## Setup

Python must satisfy `>=3.10,<3.13`; `.python-version` selects 3.10. On a new checkout, provision the locked environment with:

```powershell
uv sync --frozen --group dev
```

That setup command may download dependencies. It was **not** run during final submission preparation; the existing environment was reused. The public Git checkout includes data, tests, report and recorded evidence, so those can be reviewed without model downloads or GPU access.

Declared dependencies: torch, transformers, datasets, accelerate, sentencepiece, matplotlib, numpy, pandas, scikit-learn and ucimlrepo. Development dependencies: pytest and Ruff. The checked local environment used Python 3.10.21, uv 0.12.16, torch 2.14.1, transformers 5.18.0, datasets 4.8.5, accelerate 1.15.0, sentencepiece 0.2.2, pandas 2.3.3, numpy 2.2.6, pytest 9.1.1 and Ruff 0.16.10. The lockfile records exact resolution; hardware/platform differences can still change generated results.

## Tests and lint

From the project root, with the existing Windows environment:

```powershell
.venv/Scripts/python.exe -B -m pytest -p no:cacheprovider tests
.venv/Scripts/ruff.exe check --no-cache .
```

Portable uv equivalents after setup:

```powershell
uv run --no-sync pytest -p no:cacheprovider tests
uv run --no-sync ruff check .
```

`run_steps.ps1` now runs verification only. It does not download data, regenerate artifacts, run model inference, or retrain. Tests use synthetic fixtures and manifest-backed development data; they do not need model weights or a GPU. The final test result is recorded in [submission/SUBMISSION_STATUS.md](submission/SUBMISSION_STATUS.md).

## Assignment commands and preservation rules

Run packaged modules with `python -m`, from the directory containing `pyproject.toml`. The following are supported interfaces, **not a batch command to rerun the frozen submission**:

| Step | Actual command/interface | Important behavior |
|---|---|---|
| Download | `uv run --no-sync python -m sba_928.download_dataset` | Network access; overwrites the local source CSV. Not run for submission. |
| Explore | `uv run --no-sync python -m sba_928.explore_data` | Writes descriptive summaries/plots into local `results/`. |
| 01 | `uv run --no-sync python -m sba_928.step_01_generate_dataset` | In-memory preparation and validation only; does not rewrite frozen split files. Requires cached tokenizer. |
| 02 | `uv run --no-sync python -m sba_928.step_02_prompt_engineering --limit 12 --device cpu` | Inference with four styles; can overwrite local prompt-baseline outputs. |
| 03 | `uv run --no-sync python -m sba_928.step_03_fine_tune --epochs 3 --batch-size 2 --learning-rate 0.00005` | Retrains and writes checkpoints. Historical reference only; do not run against the archived experiment. |
| 04 | `uv run --no-sync python -m sba_928.step_04_compare_models --authorize-final-comparison` | Explicit held-out access and model inference; writes local comparison artifacts. Not rerun for submission. |
| 05 | `uv run --no-sync python -m sba_928.step_05_bias_audit --authorize-bias-audit` | Train/validation audit; refuses an existing output directory. |
| 06 | See custom example below | Writes a fresh unique custom-inference run; does not read project datasets. |

The instructor's illustrative `--config`, `--fine-tuned-model`, and customer-review flags are not accepted by these implementations. Use the actual interfaces above. Training settings are documented in the saved resolved configuration rather than a new `config/default.json`.

## Model availability and exact reproduction

The pretrained base is `google/flan-t5-small`, cached locally at revision `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab`. The final fine-tuned model is **local only** at `artifacts/models/flan-t5-sba-928/final/`.

Fine-tuned weight SHA-256:

```text
df40f62a972c0489eae1be479e237ddffad26c14928de52f0ea775eccd5cf862
```

The complete model directory needs `config.json`, `generation_config.json`, `tokenizer_config.json`, `tokenizer.json`, and `model.safetensors`. Small configuration copies are in [training evidence](submission/evidence/training/). **Weights are not in Git.** A fresh clone cannot reproduce fine-tuned inference until the matching checkpoint is restored through an instructor-approved storage method. No upload or download location has been created. Do not substitute the legacy model under `models/flan-t5-market-research/`.

Steps 05 and 06 explicitly validate Transformers 5.18.0. Cached-only use in PowerShell:

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:HF_DATASETS_OFFLINE = '1'
```

These settings prevent missing model artifacts from being downloaded; they do not make an absent checkpoint available. Installing dependencies or provisioning model caches is separate from reproducing the frozen results. The base loader's tying warnings were investigated: the installed loader preserves distinct saved output-head weights. Do not call `tie_weights()` or edit the checkpoint to silence the warnings.

## Step 06: verified custom inference

Actual interface:

```powershell
uv run --no-sync python -m sba_928.step_06_custom_inference --instruction "Summarize the supplied conversion rate and state one limitation. Use only the supplied facts." --context "Synthetic deployment smoke input. Sessions: 80; purchases: 12; conversion rate: 15.0%. Descriptive data; no causation established." --compare-original
```

Exactly one of `--context` or `--context-file` is required. The input limit is 256 tokens without truncation. Both models use the same input format, deterministic two-beam decoding, 180 new tokens and disabled caching. Each execution saves independent, unreviewed evidence to a unique directory in `artifacts/custom_inference/`.

The fresh submission smoke check **executed successfully**, loaded both local models, and verified matching 53-token inputs and generation settings. However, the fine-tuned response repeated `Convertible data: 80%.`; the base response repeated `Convertible data to a single volume.` Both reached 180 generated tokens. This is a runtime demonstration, **not successful market-research reasoning**. [Saved custom evidence](submission/evidence/custom_inference/) documents it.

The implementation accepts generic market-research instructions/context. It does not implement the instructor example's `--model --product --rating --review` flags or guarantee customer-review labels. That assignment-interface adaptation needs instructor acceptance; no new feature has been added to conceal it.

## Results and reproducibility boundaries

- Prompt experiment: 12 questions, four variants, 48 recorded and reviewed base-model responses. Elaborate prompts did not consistently improve completeness or grounding.
- Fine-tuning: 18 training examples, three validation examples, three epochs and 27 steps. Validation loss reached 3.402307; lower loss did not establish better generated answers.
- Historical held-out comparison: three records, six responses. Recorded token-F1 was 0.135834 versus 0.056171; recorded rubric means were 1.83 versus 1.25 (base/fine-tuned).
- Bias audit: five cases, ten reviewed responses from seen training/validation evidence. Four-dimension means were 2.15 versus 2.10. Protected-group discrimination cannot be assessed from the available variables.
- Development pilot: 16 synthetic prompts / 15 scenarios / 32 actual outputs. All ended with EOS. Existing strict prose scoring has 0% numerical coverage; accuracy and semantic ratings remain pending, not zero.

These evidence chains and their different sample sizes/scales must remain separate. The test set was previously inspected during diagnostics; it is not a pristine future benchmark. Fixed seeds and deterministic decoding do not guarantee bitwise reproducibility across devices or library versions. Public evidence exports are copies; original byte hashes inside metadata refer to the original local artifacts. [The export manifest](submission/evidence_manifest.json) records the public mapping and hashes. Synthetic workflow fixtures are not included as evidence of model quality.

## Submission

The repository was already initialized. Preparation creates a reviewed local commit only; no remote is configured and no push is authorized. Before uploading the GitHub link, confirm the checkpoint-delivery method and acceptance of the adapted Step 06 interface. If the original Per Scholas form is required, transfer the academic report into it; that form was not available locally. Existing validation and review limitations are disclosed in the final report rather than fabricated away.
