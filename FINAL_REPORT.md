# SBA 928: Enhancing Market Research with AI Prompt Engineering

**Course:** CAX215

**Assessment:** SBA 928

**Prepared:** October 14, 2026

**Research question:** How effectively can prompt engineering and small-scale FLAN-T5 fine-tuning support grounded analysis of online purchasing behavior?

## Abstract

This project evaluates two distinct interventions: changing instructions supplied to a pretrained language model, and updating its parameters with supervised examples. Public online-shopping data supported twelve research questions, four prompt variants per question, and twenty-four split-specific instruction–context–target records. FLAN-T5-small was fine-tuned on eighteen records for three epochs and compared with the pretrained base model on three held-out records. Five additional fairness-risk cases and a sixteen-prompt synthetic development pilot provided diagnostic evidence. The recorded prompt reviews do not show that more elaborate instructions consistently improved answers. Fine-tuning reduced validation loss slightly across epochs but did not demonstrate improved generated-answer quality. Observed failures included incomplete comparisons, number-to-group attribution errors, incorrect percentage interpretation, and repetition. These findings support careful grounding and human review rather than deployment of the model's outputs as reliable business conclusions.

## 1. Introduction

Market research requires more than fluent descriptions: an analyst must attach quantities to the correct groups, distinguish associations from causal effects, and explain uncertainty. This study tests whether a compact instruction-following model can perform those tasks using supplied shopping-session statistics.

Prompt engineering changes the instructions while keeping the model's parameters fixed. Inference generates a response. Fine-tuning changes parameters using paired source texts and targets. Keeping these activities separate makes it possible to examine the evidence behind each intervention rather than assuming a more detailed prompt or a trained checkpoint is automatically better. The six assessment requirements are addressed below, following the available local report outline. The separately referenced Per Scholas submission form was not available as an editable template in this project.

## 2. Dataset selection and preparation

The source is the public **UCI Online Shoppers Purchasing Intention Dataset**, dataset 468, credited to Sakar and Kastro and licensed CC BY 4.0 [1]. Its observational session data are relevant to conversion analysis, visitor segmentation, engagement, and timing. The local CSV contains **12,330 sessions and 18 columns**, comprising 17 predictors and the `Revenue` purchase outcome. Local descriptive analysis records **1,908 purchases and 10,422 non-purchases**, or approximately **15.5% conversion**. Boolean fields are normalized explicitly, and source validation checks required fields and values.

Relevant variables include `VisitorType`, `Month`, `Weekend`, `TrafficType`, `Region`, `Browser`, `BounceRates`, `ExitRates`, `PageValues`, and product-page engagement. Region and traffic codes have no verified real-world mappings in this analysis. Detailed protected attributes, motivations, and competitor-company performance are unavailable. Consequently, the data can support descriptive comparisons and hypotheses, but cannot justify invented demographic explanations, named competitor rankings, or causal claims.

The local exploratory results show returning visitors at approximately 13.9% conversion and new visitors at 24.9%. These are full-source descriptive statistics, not controlled estimates of visitor-type effects. Purchasing sessions also have lower mean bounce and exit rates and greater product-page engagement. Figures and the original descriptive summary are provided in [dataset evidence](submission/evidence/dataset_summary/).

### Split and provenance procedure

Source rows are normalized to typed canonical representations and hashed. Exact duplicate rows remain in the same provenance group. A seeded, outcome-stratified group assignment creates approximately 80/10/10 source partitions. Supervised examples are then constructed from **partition-local statistics**, with factual target responses and provenance identifiers. This separates source-row splitting from the later allocation of research tasks.

| Partition | Source sessions | Supervised records |
|---|---:|---:|
| Training | 9,864 | 18 |
| Validation | 1,233 | 3 |
| Test | 1,233 | 3 |
| Total | 12,330 | 24 |

The frozen [manifest](data/processed/manifest.json) records source and output hashes, unique record IDs, source-group membership, token lengths, and passing exact/near-duplicate checks. The current preparation script is a **dry-run validator**; it does not rewrite these files. Legacy records in `prompts/train.jsonl`, `prompts/test.jsonl`, and earlier one-epoch results are not the evidence used for this report.

## 3. Requirement 1: diverse market-research prompts

The [prompt catalog](prompts/prompt_catalog.json) contains twelve research questions. Their breadth connects the assessment to multiple analytical tasks without inventing variables absent from the source.

| ID | Research area | Basic research question |
|---|---|---|
| 1 | Consumer behavior | Analyze purchasing-session behavior. |
| 2 | Visitor segmentation | Compare new and returning visitors. |
| 3 | Market trends | Identify observed monthly purchasing patterns. |
| 4 | Traffic analysis | Compare coded traffic groups. |
| 5 | Engagement | Compare bounce rates by purchase outcome. |
| 6 | Engagement | Compare exit rates by purchase outcome. |
| 7 | Timing | Compare weekend and non-weekend sessions. |
| 8 | Technology | Examine browser-category purchasing patterns. |
| 9 | Regional analysis | Compare coded regions. |
| 10 | Purchase intention | Examine page value and purchasing. |
| 11 | Marketing strategy | Propose conversion hypotheses for testing. |
| 12 | Competitive implications | Discuss capabilities without claiming competitor rankings. |

The competitive question is deliberately limited to possible organizational capabilities. The dataset does not contain named competitor outcomes, so a responsible answer must state that limitation.

## 4. Requirement 2: prompt variations and results

Each question has **basic**, **specific**, **role/context**, and **grounded/constrained** variants. The [variation document](prompts/prompt_variations.md) contains all forty-eight prompts. For example, a basic bounce-rate request asks for analysis; the specific version identifies the two purchase groups; the role version asks an analyst for a testable business hypothesis; and the grounded version restricts conclusions to supplied averages and prohibits unsupported causal claims.

The saved experiment generated **48 pretrained-model responses: 12 cases × 4 variants**, using shared case evidence, a 512-token input limit, and deterministic two-beam decoding with at most 180 new tokens. No parameters were trained in this phase. The variants sometimes change requested scope as well as wording, so the comparison does not isolate wording alone. Moreover, prompt-engineering cases use full-source aggregates: their distinct record IDs do not establish source-row independence from the later supervised partitions.

Recorded, user-approved human reviews use a **1–5 scale**. The following averages are copied from the saved summary, not newly assigned ratings.

| Variant | Relevance | Completeness | Grounding | Specificity |
|---|---:|---:|---:|---:|
| Basic | 2.17 | 1.33 | 3.83 | 2.58 |
| Specific | 2.08 | 1.25 | 3.75 | 2.08 |
| Role/context | 1.75 | 1.17 | 3.17 | 2.08 |
| Grounded/constrained | 2.00 | 1.25 | 3.25 | 2.08 |

Format adherence was applicable to only six responses across the variants; its averages are not directly comparable. In the bounce-rate case, the basic and specific outputs supplied only the purchasing group's mean, while the grounded output gave an incomplete and slightly altered number. All four weekend variants returned the same single rate, omitting the requested comparison. The competitor case correctly acknowledged missing competitor fields but did not consistently provide useful implications. Correct copied fragments can be grounded while remaining incomplete.

Thus, constraints expressed the intended analytical standard but did not reliably produce it. Greater prompt detail was not consistently beneficial in these recorded results. Full responses, review notes, and the rubric are in [prompt-engineering evidence](submission/evidence/prompt_engineering/).

## 5. Requirement 3: FLAN-T5 fine-tuning

`google/flan-t5-small` is a pretrained, instruction-tuned text-to-text model, not an untrained model [2, 3]. Its relatively small size made a genuine local CPU experiment feasible. The diagnostic loader reports approximately 77 million parameters. The experiment trained it to produce analyst-style responses grounded in the supplied statistics.

Inputs use `instruction + "\n\nContext:\n" + context`. Source and target tokenization are separate. Preflight checks reject sources above **256 tokens** or targets above **96 tokens** without truncating them. The sequence-to-sequence collator pads targets with `-100`, excluding padding from loss. Only training records update parameters; validation loss selects the checkpoint. The training script does not open held-out test records.

| Setting or result | Recorded value |
|---|---|
| Epochs / optimizer steps | 3 / 27 |
| Training / validation examples | 18 / 3 |
| Training batch size | 2 |
| Learning rate | 0.00005 |
| Weight decay | 0.01 |
| Optimizer / scheduler | `adamw_torch_fused` / linear |
| Seed / data seed | 42 / 42 |
| Hardware / mixed precision | CPU / fp16 disabled |
| Recorded training runtime | 46.5802 seconds |
| Aggregate training loss | 4.241846 |
| Validation losses, epochs 1–3 | 3.436954; 3.414304; 3.402307 |
| Best checkpoint | `checkpoint-27` |

Training evidence is in [the resolved configuration](submission/evidence/training/training_config_resolved.json) and [training summary](submission/evidence/training/training_summary.json). The final checkpoint remains locally at `artifacts/models/flan-t5-sba-928/final/`. Its weight SHA-256 is `df40f62a972c0489eae1be479e237ddffad26c14928de52f0ea775eccd5cf862`. Large weights and optimizer states are excluded from Git; the checkpoint location, hash, and small configuration files are documented. Exact inference from a fresh clone requires separately restoring that checkpoint through an instructor-approved method.

The decline in validation loss demonstrates optimization of the supervised objective, not improved numerical reasoning or business-analysis accuracy. Eighteen examples and three validation records provide limited task and numeric diversity. Training largely repeats supplied rates rather than teaching worked division or percentage-point subtraction. No additional fine-tuning was performed during submission preparation.

## 6. Requirement 4: pretrained versus fine-tuned model

### Recorded held-out comparison

The saved comparison used three held-out records, identical formatted inputs and tokenization, and deterministic settings: `do_sample=false`, `num_beams=2`, `max_new_tokens=180`, and `use_cache=false`. This report reuses existing evidence; it does not rerun held-out inference.

| Recorded measure | Pretrained base | Fine-tuned |
|---|---:|---:|
| Supplemental token-overlap F1 | 0.135834 | 0.056171 |
| Reviewed relevance, 1–5 | 2.33 | 2.00 |
| Reviewed completeness, 1–5 | 1.33 | 1.00 |
| Reviewed grounding, 1–5 | 1.67 | 1.00 |
| Reviewed specificity, 1–5 | 2.00 | 1.00 |
| Mean of four applicable rubric dimensions | 1.83 | 1.25 |

Format adherence was not applicable because no particular format was required. Token-F1 measures lexical overlap, not factual accuracy. The recorded review identifies incomplete regional analysis, a numeric corruption, an unexplained short answer, and a repetitive fine-tuned response to the timing task. The base model performed somewhat better on these three records, but neither completed all requested tasks. The sample is too small to support population-wide ranking. [Saved comparison evidence](submission/evidence/model_comparison/) includes outputs and reviewer rationales.

### Compatibility investigation

The original saved configuration declares `tie_word_embeddings=false`; the fine-tuned configuration declares `true`. Transformers 5.18.0 normalizes the flag, but its loader detects unequal saved input/output-head tensors and leaves them separate. Both loaded models preserve their saved values and disable decoder-output scaling. Therefore, the flag discrepancy is a compatibility limitation, but was **not demonstrated as the cause** of poor performance. No weights or configurations were repaired.

### Controlled development pilot

A separate **16-prompt synthetic pilot across 15 scenarios** tested conversion, percentage points, coded regions/traffic, boolean groups, thresholds, causation, fairness, and recommendations. An intentional recommendation paraphrase pair shares one scenario. It is development evidence, not a fresh final benchmark. Inputs were 53–87 tokens, both tokenizers matched, and the same historical decoding settings were used for both models.

| Pilot measure | Pretrained base | Fine-tuned |
|---|---:|---:|
| Responses | 16 | 16 |
| EOS endings / token-cap endings | 16 / 0 | 16 / 0 |
| Mean new tokens | 13.75 | 23.25 |
| Mean word-trigram repetition, per prompt | 2.41% | 2.18% |
| Mean repetition, equally weighted scenarios | 2.57% | 2.32% |
| Duplicate-sentence responses | 0 | 0 |
| Automated numerical coverage | 0/56 | 0/56 |
| Automated arithmetic coverage | 0/9 | 0/9 |

The strict scorer supports explicit fact-line attribution; actual outputs were prose or short labels. Accordingly, numerical and arithmetic accuracy remain unavailable: **zero coverage is not zero accuracy**. These outputs require manual annotation, and no final pilot ratings have been invented.

Direct qualitative checks nevertheless reveal failures. Both models returned `0.064%` where the supplied fraction corresponds to **6.4%**, selected ineligible **G99** rather than **G100**, changed Banner A's **18.0%** to **10.0%**, and mislabeled the Large segment as Small. For a required controlled layout test, the outputs were `A` and `Layout A`. The fine-tuned model often copied more context but did not reliably calculate results or complete instructions. [Pilot evidence](submission/evidence/development_pilot/) preserves raw responses, scenario summaries, and an uncompleted review worksheet.

## 7. Requirement 5: bias and fairness evaluation

The audit examines five risks: demographic assumptions, invented coded labels, group attribution, small-sample interpretation, and association versus causation. Its five cases produce ten responses and reuse training/validation evidence. This makes it a diagnostic risk audit, not an independent generalization benchmark or a protected-group fairness measurement.

| Recorded 1–5 review average | Pretrained base | Fine-tuned |
|---|---:|---:|
| Relevance | 2.40 | 2.80 |
| Completeness | 1.80 | 1.60 |
| Grounding | 2.40 | 2.20 |
| Specificity | 2.00 | 1.80 |
| Four-dimension mean | 2.15 | 2.10 |

The fine-tuned model better stated the absence of demographic fields in one case. Other outputs repeated phrases or omitted group comparisons. Near-equal overall means do not demonstrate improved fairness. All ten recorded response reviews, excerpts, and rationales are preserved in [bias-audit evidence](submission/evidence/bias_audit/).

Outcome imbalance may hide failures affecting purchasing sessions if analysis emphasizes majority behavior. Unequal segment sizes also make some headline comparisons fragile: the training `Other` visitor group has only 62 sessions, compared with 8,423 returning visitors. A minimum reporting threshold is a policy safeguard, not a significance test. Geography and traffic codes must not be converted into unverified names or demographic identities.

The observational dataset is historically bounded and cannot establish present-day shopping behavior or causal effects. Language-model pretraining may introduce biases beyond this small audit; those biases were not independently isolated here. No verified protected attributes or group-level classification predictions are available, so demographic parity, equalized odds, or discrimination findings cannot be supported. Incorrect or incomplete business analysis creates **potential fairness risk**, which is distinct from demonstrated protected-group discrimination.

## 8. Requirement 6: fairness strategies

Responsible use would require reporting both purchasing and non-purchasing outcomes, presenting segment counts alongside rates, retaining small groups while flagging uncertainty, and keeping coded labels unchanged unless documentation supplies a mapping. Prompts should require source-grounded claims and distinguish proposed business hypotheses from observed findings.

Every recommendation should identify what would be tested, the outcome to measure, and relevant conditions to hold comparable. For example, an observed conversion difference can motivate a controlled experiment but does not justify guaranteed gains or automatic budget shifts. Numeric review must check the denominator, group attribution, units, and the difference between percentage points and relative percentage change.

Human review should compare claims against the actual evidence, including accurate-looking fragments that omit essential analysis. A blinded second review of a prespecified subset would help detect inconsistent judgments. Reassessment with newer, appropriately governed data is necessary before business use. These are recommendations, not claims that the current model meets the proposed standard.

## 9. Step 06 verification, limitations, and conclusion

### Actual custom-inference status

The implemented Step 06 interface accepts `--instruction`, exactly one of `--context` or `--context-file`, and optionally `--compare-original`. A fresh offline smoke run on October 14, 2026 loaded both local models, processed the same 53-token synthetic input, generated responses, and saved separate provenance and warnings. It successfully checked matching generation settings and unchanged fine-tuned file hashes.

The input supplied **80 sessions, 12 purchases, and 15.0% conversion**. The fine-tuned model repeatedly emitted `Convertible data: 80%.`; the base model repeatedly emitted `Convertible data to a single volume.` Both reached 180 generated tokens. Thus, the **mechanism executes, but useful and grounded custom analysis was not demonstrated**. Outputs remain unreviewed, and successful execution must not be described as successful task performance. [Custom-inference evidence](submission/evidence/custom_inference/) records the exact input, raw outputs, metadata, and warnings.

The instructor's example uses `--model`, `--product`, `--rating`, and `--review`, with customer-review labels. Those options and that response schema are not implemented here; this project adapts Step 06 to the UCI session-analysis domain. Acceptance of that adaptation is a submission consideration. No new CLI features or training were added to hide the gap.

### Limitations

The supervised dataset and validation sample are very small; task allocation and numeric variation are narrow. Prompt styles vary scope, full-source prompt aggregates overlap supervised source populations, and the original three-case test has subsequently been inspected during diagnostics. It should not be treated as a pristine benchmark for future model selection. The synthetic pilot broadens task coverage but supplies neither representative market samples nor completed semantic ratings.

Recorded rubric scores are descriptive human judgments without reported inter-reviewer agreement. The 1–5 historical rubric must not be mixed with proposed 0–2 future review ratings. The automatic scorer has no reliable general prose attribution; arithmetic-result correctness alone would not prove reasoning. Deterministic decoding and fixed seeds do not guarantee bitwise identity across hardware or library versions. Finally, large checkpoint files are local rather than included in the Git repository, so restoring them is necessary for exact inference replication.

### Conclusion

The project demonstrates a complete local experimental chain: public-source preparation, diverse prompts, actual parameter updates, comparison with a pretrained base model, and a documented fairness-risk audit. It does **not** demonstrate that this small fine-tuning experiment improved market-research answer quality. More elaborate prompts also did not consistently improve the recorded responses. Lower validation loss, longer answers, and fewer repeated n-grams are insufficient evidence of better analytical performance.

The defensible conclusion is that this model configuration requires human verification for numerical and business reasoning. Any future work should begin with auditable response review and stronger independent evaluation, rather than assuming further fine-tuning will solve the observed problems. Development has ended for this submission; no additional evaluation features or retraining are claimed.

## 10. References and evidence

1. Sakar, C., & Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset*. UCI Machine Learning Repository. [Dataset DOI and attribution](https://doi.org/10.24432/C5F88Q).
2. Google / Hugging Face. *FLAN-T5-small model card*. [Model description and limitations](https://huggingface.co/google/flan-t5-small).
3. Chung, H. W., et al. (2022). *Scaling Instruction-Finetuned Language Models*. arXiv:2210.11416. [Research paper](https://arxiv.org/abs/2210.11416).
4. Project-local instructor requirements, technical clarifications, and learner guide, reviewed for assignment alignment. Raw class-message copies are excluded from the public repository.

All experiment values above come from existing saved artifacts or the explicitly identified custom-inference smoke check. Public evidence copies redact local filesystem paths; raw response text and numeric results are preserved. [The export manifest](submission/evidence_manifest.json) records original and public-file hashes. Historical relative paths/hashes inside copied metadata may refer to local archives; use the manifest to locate their public counterparts. See [README.md](README.md) for commands and reproducibility boundaries.
