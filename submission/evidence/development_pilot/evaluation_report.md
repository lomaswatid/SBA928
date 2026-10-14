# SBA-928 Phase 4 controlled development evaluation

**Completed:** both local models generated responses for all 16 development prompts under identical deterministic conditions. No held-out examples or predictions were accessed.

## Preflight

- Pilot/source-manifest hashes match the frozen Phase 2 artifacts; provenance and nine calculation checks pass.
- Sixteen explicitly synthetic cases represent fifteen scenarios; one intentional layout paraphrase pair remains grouped.
- Both models/tokenizers load locally. Token IDs and attention masks match for every input; lengths are 53?87 tokens, below 256.
- Input format: `instruction + "\n\nContext:\n" + context`, without truncation or provenance/references.
- CPU, float32, four threads, seed 42, evaluation mode and inference mode.
- Settings: `do_sample=false`, `num_beams=2`, `max_new_tokens=180`, `use_cache=false`. Full effective settings match; EOS/start/pad tokens match.
- Python 3.10.21; PyTorch 2.14.1; Transformers 5.18.0; SentencePiece 0.2.2. Complete hashes and versions are in [preflight.json](preflight.json).
- Both output heads remain separate, match their saved weights, and retain disabled decoder-output scaling. All parameters are finite.
- Model and dataset hashes remain unchanged after generation. No checkpoints or source files were modified.

**Weight SHA-256:**

- Original: `495fa51e204676f1a857a9fc13c4c89f3f5ba9f480b898cebca02add25e6d749`.
- Fine-tuned: `df40f62a972c0489eae1be479e237ddffad26c14928de52f0ea775eccd5cf862`.

## Model comparison

| Measure | Original | Fine-tuned |
|---|---:|---:|
| Responses / scenarios | 16 / 15 | 16 / 15 |
| EOS endings | 16 | 16 |
| Token-cap endings | 0 | 0 |
| Mean new tokens (including EOS) | 13.75 | 23.25 |
| Mean word-trigram repetition, per case | 2.41% | 2.18% |
| Mean word-trigram repetition, equally weighted scenarios | 2.57% | 2.32% |
| Duplicate-sentence responses | 0 | 0 |
| Automated numerical coverage | 0 / 56 | 0 / 56 |
| Automated arithmetic coverage | 0 / 9 | 0 / 9 |
| Automated numerical/arithmetic accuracy | Not available | Not available |

These repetition fractions are mechanical signals, not quality scores. The fine-tuned responses are longer and often copy more supplied facts; this does not establish improved task completion.

## Automated scoring and manual review

All 32 actual outputs are prose or short labels, not the Phase 3 canonical fact-line format. They are retained as `free_text`. Therefore all 112 factual expectations and 18 calculation results require review. **Zero coverage is not zero accuracy.** No numerical accuracy is claimed or computed from an unreliable parser.

The existing offline scorer consumed the saved response matrix successfully. [Its report](scoring/report.json) preserves individual outputs, generation metadata, dimension denominators and pending statuses. Its `models_executed=false` flag refers to the scoring stage only; [inference_manifest.json](inference_manifest.json) records the actual controlled inference.

[manual_review.csv](manual_review.csv) contains 32 response rows and four pending semantic dimensions each (128 judgments), with expected facts, calculations, requirements and evidence/rationale columns. See [manual_review_guide.md](manual_review_guide.md).

[scenario_summary.json](scenario_summary.json) contains 15 scenario summaries per model. The layout variants are averaged within one scenario before equal weighting across scenarios. No independent-scenario accuracy or significance estimate is claimed.

## Representative actual outputs

| Task and expected result | Original | Fine-tuned |
|---|---|---|
| Fraction 0.064 means 6.4% | `0.064%` | `0.064%` |
| Spring minus Winter = 5.5 percentage points | `0.2%` | `Spring minus Winter in percentage points, not relative percent change.` |
| Eligible headline winner is G100; G99 has only 99 sessions | `G99` | `G99` |
| Propose a controlled layout experiment | `A` | `Layout A` |

Both models also change Banner A from the supplied 18.0% to 10.0%, and label the 400-session Large segment as Small. These are direct qualitative checks against supplied synthetic facts, not completed rubric ratings or aggregate accuracy scores. Fine-tuned conversion-01 copies the counts and formula but does not calculate the requested 17.2%.

Full exact outputs and case expectations are in [response_comparison.md](response_comparison.md) and [saved_responses.jsonl](saved_responses.jsonl).

## Runtime and limitations

- No runtime errors; the controlled driver completed in about 49.3 seconds, excluding the separate initial preflight.
- Captured warnings: two known output-head tying warnings plus 32 notices that `max_new_tokens=180` takes precedence over the compatible `max_length=181`. No manual repair was performed.
- No token-cap or duplicate-sentence loops occurred in this pilot. Short/incomplete outputs and factual errors still occurred.
- Sixteen synthetic prompts across fifteen scenarios are a small development pilot, not a representative real-world or final-test benchmark.
- Semantic and numerical prose reviews remain pending. No dependable model ranking follows from automated scores with zero coverage.
- Deterministic decoding does not guarantee bitwise identity across library versions or hardware.

## Recommended next steps

1. Complete evidence-based manual reviews, preferably blinded and with a double-reviewed subset.
2. Approve an auditable prose annotation/review-import workflow before computing numerical accuracy.
3. Compare reviewed results per scenario before considering additional fine-tuning or parser/prompt changes.
4. Freeze these outputs and keep development selection separate from a future untouched final benchmark.

No further changes have been implemented. Phase 4 is complete; additional work awaits approval.
