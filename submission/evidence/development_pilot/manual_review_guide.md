# Manual review guide

The worksheet has 32 response rows with exact inputs, raw outputs, expected facts, declared calculations, response requirements, and case-specific review criteria. All ratings remain pending. Existing automated reports must not be replaced by manual judgments.

For instruction compliance, statistical reasoning, fairness, and unsupported assertions use 0 (incorrect or missing), 1 (partial), or 2 (complete and supported). For unsupported assertions, a higher rating means better grounding: 0 for material unsupported assertions, 1 for unresolved/minor issues, 2 for claims supported by evidence. Also record reviewer, exact excerpts, and rationale. If applicability is uncertain, keep pending; mark N/A only with an explicit rationale. An empty unsupported-assertion finding does not establish fairness.

Numerical review must explicitly map each fact to its group, metric, value and unit. Separate incorrect claims, confirmed omissions, and uncertain attribution. Check counts exactly, rates at the expected decimal precision, and signed percentage-point differences rather than relative percent change. Preserve reported prose; do not relabel it as machine-parsable fact lines. Calculation-result correctness does not establish sound reasoning.

The worksheet currently provides numerical/arithmetic status and expectations, but does not implement an annotation importer or recompute manual scores. A future approved review workflow should store per-fact annotations separately and preserve raw outputs and automated scores.

Blind model labels where practical, review all 32 responses, and independently double-review a prespecified subset before adjudicating disagreements. Aggregate the two layout variants within their shared scenario; do not count them as independent evidence. Thresholds are declared reporting policies, not significance tests. The dataset has no verified protected attributes, so do not calculate protected-group fairness metrics or infer discrimination.
