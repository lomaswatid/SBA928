# Actual development responses

All outputs below are actual local FLAN-T5 generations. There are 16 prompts, 15 scenarios and 32 outputs. The layout-recommendation pair is one scenario. No semantic ratings are assigned here.

## pilot-conversion-01

Scenario: `pilot-conversion-01`; family: `conversion`.

**Instruction:** Calculate the purchase conversion rate and state its denominator.

**Context:**

> Synthetic scenario; invented values, not observations from the UCI dataset.
> Store A recorded 250 sessions and 43 purchases. Conversion means purchases divided by sessions, expressed as a percentage.

**Expected facts:**

| Group | Metric | Value | Unit |
|---|---|---:|---|
| Store A | sessions | 250 | count |
| Store A | purchases | 43 | count |
| Store A | conversion_rate | 17.2 | percent |

**Response requirements:**

- Report 17.2% and the 250-session denominator.
- Use sessions rather than non-purchases as the denominator.

**original** (4 new tokens; eos):

> Purchase conversion rate

**fine_tuned** (24 new tokens; eos):

> Store A recorded 250 sessions and 43 purchases. Conversion means purchases divided by sessions, expressed as a percentage.

## pilot-conversion-02

Scenario: `pilot-conversion-02`; family: `conversion`.

**Instruction:** Interpret the rate given as a fraction and check it against the purchase counts.

**Context:**

> Synthetic scenario; invented values, not observations from the UCI dataset.
> Store B recorded 250 sessions and 16 purchases. The supplied purchase-rate fraction is 0.064, not 0.064%.

**Expected facts:**

| Group | Metric | Value | Unit |
|---|---|---:|---|
| Store B | purchase_rate_fraction | 0.064 | fraction |
| Store B | conversion_rate | 6.4 | percent |

**Response requirements:**

- Explain that fraction 0.064 corresponds to 6.4%.
- Check the result against 16 purchases in 250 sessions.

**original** (4 new tokens; eos):

> 0.064%

**fine_tuned** (4 new tokens; eos):

> 0.064%

## pilot-percentage-01

Scenario: `pilot-percentage-01`; family: `percentage_points`.

**Instruction:** Calculate Spring minus Winter in percentage points, not relative percent change.

**Context:**

> Synthetic scenario; invented values, not observations from the UCI dataset.
> Spring conversion is 23.7%; Winter conversion is 18.2%. Both are already percentages.

**Expected facts:**

| Group | Metric | Value | Unit |
|---|---|---:|---|
| Spring | conversion_rate | 23.7 | percent |
| Winter | conversion_rate | 18.2 | percent |
| Spring minus Winter | conversion_difference | 5.5 | percentage_point |

**Response requirements:**

- State a positive 5.5-percentage-point difference.
- Preserve the requested subtraction direction.

**original** (3 new tokens; eos):

> 0.2%

**fine_tuned** (14 new tokens; eos):

> Spring minus Winter in percentage points, not relative percent change.

## pilot-percentage-02

Scenario: `pilot-percentage-02`; family: `percentage_points`.

**Instruction:** Compute Experiment minus Baseline and interpret the sign.

**Context:**

> Synthetic scenario; invented values, not observations from the UCI dataset.
> Experiment purchase conversion is 12.4%; Baseline purchase conversion is 16.9%. No causal study design is supplied.

**Expected facts:**

| Group | Metric | Value | Unit |
|---|---|---:|---|
| Experiment | conversion_rate | 12.4 | percent |
| Baseline | conversion_rate | 16.9 | percent |
| Experiment minus Baseline | conversion_difference | -4.5 | percentage_point |

**Response requirements:**

- Report -4.5 percentage points, or explicitly a 4.5-point decrease.
- Do not infer that the experiment caused the difference.

**original** (17 new tokens; eos):

> Experiment purchase conversion is 12.4%; Baseline purchase conversion is 16.9%

**fine_tuned** (25 new tokens; eos):

> Experiment purchase conversion is 12.4%; Baseline purchase conversion is 16.9%. No causal study design is supplied.

## pilot-region-01

Scenario: `pilot-region-01`; family: `region_traffic`.

**Instruction:** Identify the highest and lowest eligible Region codes; do not invent place names.

**Context:**

> Synthetic scenario; invented values, not observations from the UCI dataset.
> Region R11: 240 sessions, 15.0% conversion. Region R22: 150 sessions, 20.0%. Region R33: 24 sessions, 37.5%. Eligibility is at least 100 sessions. Codes have no geographic mappings.

**Expected facts:**

| Group | Metric | Value | Unit |
|---|---|---:|---|
| Region R22 | conversion_rate | 20.0 | percent |
| Region R22 | sessions | 150 | count |
| Region R11 | conversion_rate | 15.0 | percent |
| Region R11 | sessions | 240 | count |

**Response requirements:**

- Select R22 as highest and R11 as lowest among eligible groups.
- Retain R33 for visibility and flag its small sample.

**original** (4 new tokens; eos):

> Region R11

**fine_tuned** (52 new tokens; eos):

> Region R11: 240 sessions, 15.0% conversion. Region R22: 150 sessions, 20.0%. Region R33: 24 sessions, 37.5%. Eligibility is at least 100 sessions. Codes have no geographic mappings.

## pilot-traffic-01

Scenario: `pilot-traffic-01`; family: `region_traffic`.

**Instruction:** Compare the supplied traffic codes using counts and conversion; keep the code labels.

**Context:**

> Synthetic scenario; invented values, not observations from the UCI dataset.
> Traffic T41: 180 sessions, 15.0% conversion. Traffic T52: 260 sessions, 10.0% conversion. No verified channel-name mappings or controlled effects exist.

**Expected facts:**

| Group | Metric | Value | Unit |
|---|---|---:|---|
| Traffic T41 | sessions | 180 | count |
| Traffic T52 | sessions | 260 | count |
| Traffic T41 | conversion_rate | 15.0 | percent |
| Traffic T52 | conversion_rate | 10.0 | percent |
| Traffic T41 minus T52 | conversion_difference | 5.0 | percentage_point |

**Response requirements:**

- Report both counts and rates, and a 5.0-point T41-minus-T52 difference.
- Do not rename codes as paid search, email or social channels.

**original** (24 new tokens; eos):

> Traffic T41: 180 sessions, 15.0% conversion. Traffic T52: 260 sessions, 10.0% conversion.

**fine_tuned** (36 new tokens; eos):

> Traffic T41: 180 sessions, 15.0% conversion. Traffic T52: 260 sessions, 10.0% conversion. No verified channel-name mappings or controlled effects exist.

## pilot-boolean-01

Scenario: `pilot-boolean-01`; family: `boolean_groups`.

**Instruction:** Compare promotion and non-promotion sessions while keeping boolean labels attached to the right groups.

**Context:**

> Synthetic scenario; invented values, not observations from the UCI dataset.
> Promo=True means a promotion was shown: 160 sessions, 20.0% conversion. Promo=False means no promotion: 240 sessions, 10.0% conversion. Assignment was observational.

**Expected facts:**

| Group | Metric | Value | Unit |
|---|---|---:|---|
| Promo=True | sessions | 160 | count |
| Promo=False | sessions | 240 | count |
| Promo=True | conversion_rate | 20.0 | percent |
| Promo=False | conversion_rate | 10.0 | percent |
| Promo=True minus Promo=False | conversion_difference | 10.0 | percentage_point |

**Response requirements:**

- Attach 20.0% to True and 10.0% to False.
- State the 10.0-point descriptive difference without a causal claim.

**original** (42 new tokens; eos):

> Promo=True means a promotion was shown: 160 sessions, 20.0% conversion. Promo=False means no promotion: 240 sessions, 10.0% conversion. Assignment was observational.

**fine_tuned** (42 new tokens; eos):

> Promo=True means a promotion was shown: 160 sessions, 20.0% conversion. Promo=False means no promotion: 240 sessions, 10.0% conversion. Assignment was observational.

## pilot-boolean-02

Scenario: `pilot-boolean-02`; family: `boolean_groups`.

**Instruction:** Compare near-holiday and other sessions and state which group has the lower observed rate.

**Context:**

> Synthetic scenario; invented values, not observations from the UCI dataset.
> NearHoliday=True defines the near-holiday group: 80 sessions, 5.0% conversion. NearHoliday=False defines other sessions: 320 sessions, 15.0% conversion. These are descriptive statistics.

**Expected facts:**

| Group | Metric | Value | Unit |
|---|---|---:|---|
| NearHoliday=True | sessions | 80 | count |
| NearHoliday=False | sessions | 320 | count |
| NearHoliday=True | conversion_rate | 5.0 | percent |
| NearHoliday=False | conversion_rate | 15.0 | percent |
| NearHoliday=True minus NearHoliday=False | conversion_difference | -10.0 | percentage_point |

**Response requirements:**

- Identify True as lower and report both rates and counts.
- Flag that the True group is smaller; do not assert statistical significance.

**original** (6 new tokens; eos):

> Near Holiday=True

**fine_tuned** (44 new tokens; eos):

> NearHoliday=True defines the near-holiday group: 80 sessions, 5.0% conversion. NearHoliday=False defines other sessions: 320 sessions, 15.0% conversion.

## pilot-threshold-01

Scenario: `pilot-threshold-01`; family: `sample_thresholds`.

**Instruction:** Apply the declared threshold and choose the highest eligible group.

**Context:**

> Synthetic scenario; invented values, not observations from the UCI dataset.
> Minimum headline size: 100 sessions, inclusive. G99 has 99 sessions and 33.3% conversion; G100 has 100 sessions and 20.0%; G101 has 101 sessions and 9.9%.

**Expected facts:**

| Group | Metric | Value | Unit |
|---|---|---:|---|
| policy | minimum_headline_sessions | 100 | count |
| G100 | sessions | 100 | count |
| G100 | conversion_rate | 20.0 | percent |
| G99 | sessions | 99 | count |

**Response requirements:**

- Choose G100, including the exactly-at-threshold group.
- Retain and flag G99 rather than making it the headline winner.

**original** (3 new tokens; eos):

> G99

**fine_tuned** (3 new tokens; eos):

> G99

## pilot-threshold-02

Scenario: `pilot-threshold-02`; family: `sample_thresholds`.

**Instruction:** State whether any group qualifies for headline comparison; retain smaller groups.

**Context:**

> Synthetic scenario; invented values, not observations from the UCI dataset.
> Eligibility requires at least 100 sessions. Tiny1 has 26 sessions and 23.1% conversion; Tiny2 has 74 sessions and 16.2%.

**Expected facts:**

| Group | Metric | Value | Unit |
|---|---|---:|---|
| policy | minimum_headline_sessions | 100 | count |
| Tiny1 | sessions | 26 | count |
| Tiny2 | sessions | 74 | count |

**Response requirements:**

- State that no group qualifies.
- Report both small groups without selecting an eligible winner.

**original** (25 new tokens; eos):

> Tiny1 has 26 sessions and 23.1% conversion; Tiny2 has 74 sessions and 16.2% conversion.

**fine_tuned** (25 new tokens; eos):

> Tiny1 has 26 sessions and 23.1% conversion; Tiny2 has 74 sessions and 16.2% conversion.

## pilot-association-01

Scenario: `pilot-association-01`; family: `association_causation`.

**Instruction:** Explain what the banner comparison supports and what it cannot establish.

**Context:**

> Synthetic scenario; invented values, not observations from the UCI dataset.
> Banner A cohort: 300 sessions, 18.0% conversion. Banner B cohort: 300 sessions, 10.0%. Audiences differed and assignment was not randomized. No adjusted effect or significance test is supplied.

**Expected facts:**

| Group | Metric | Value | Unit |
|---|---|---:|---|
| Banner A | conversion_rate | 18.0 | percent |
| Banner B | conversion_rate | 10.0 | percent |
| Banner A minus Banner B | conversion_difference | 8.0 | percentage_point |

**Response requirements:**

- Report the 8.0-point observed difference.
- Explain that differing audiences confound a causal interpretation.

**original** (23 new tokens; eos):

> Banner A cohort: 300 sessions, 10.0% conversion. Banner B cohort: 300 sessions, 10.0% conversion.

**fine_tuned** (23 new tokens; eos):

> Banner A cohort: 300 sessions, 10.0% conversion. Banner B cohort: 300 sessions, 10.0% conversion.

## pilot-association-02

Scenario: `pilot-association-02`; family: `association_causation`.

**Instruction:** Assess the claim that Device A causes higher conversion, using only the evidence.

**Context:**

> Synthetic scenario; invented values, not observations from the UCI dataset.
> Device A: 120 sessions, 20.0% conversion. Device B: 480 sessions, 10.0%. This is an unadjusted observational comparison; customer mix and timing were not controlled.

**Expected facts:**

| Group | Metric | Value | Unit |
|---|---|---:|---|
| Device A | sessions | 120 | count |
| Device B | sessions | 480 | count |
| Device A | conversion_rate | 20.0 | percent |
| Device B | conversion_rate | 10.0 | percent |

**Response requirements:**

- Reject the causal conclusion while acknowledging the observed difference.
- Mention uncontrolled customer mix or timing.

**original** (23 new tokens; eos):

> Device A is an unadjusted observational comparison; customer mix and timing were not controlled.

**fine_tuned** (23 new tokens; eos):

> Device A is an unadjusted observational comparison; customer mix and timing were not controlled.

## pilot-fairness-01

Scenario: `pilot-fairness-01`; family: `bias_fairness`.

**Instruction:** Describe the outcome balance and which demographic fairness conclusions are unavailable.

**Context:**

> Synthetic scenario; invented values, not observations from the UCI dataset.
> Available fields are campaign code, session count and purchase outcome. No race, income or other verified protected attributes are supplied. There are 560 sessions, 84 purchases and 476 non-purchases.

**Expected facts:**

| Group | Metric | Value | Unit |
|---|---|---:|---|
| All sessions | sessions | 560 | count |
| All sessions | purchases | 84 | count |
| All sessions | non_purchases | 476 | count |
| All sessions | conversion_rate | 15.0 | percent |

**Response requirements:**

- Report 15.0% purchase conversion and the majority non-purchase outcome.
- State that protected-group fairness metrics cannot be assessed from these fields.

**original** (7 new tokens; eos):

> 476 non-purchases

**fine_tuned** (18 new tokens; eos):

> There are 560 sessions, 84 purchases and 476 non-purchases.

## pilot-fairness-02

Scenario: `pilot-fairness-02`; family: `bias_fairness`.

**Instruction:** Compare segments responsibly and explain why a higher small-group rate is not enough to declare superiority.

**Context:**

> Synthetic scenario; invented values, not observations from the UCI dataset.
> Large segment: 400 sessions, 10.0% conversion. Small segment: 20 sessions, 30.0%. Protected attributes and uncertainty estimates are unavailable.

**Expected facts:**

| Group | Metric | Value | Unit |
|---|---|---:|---|
| Large | sessions | 400 | count |
| Small | sessions | 20 | count |
| Large | conversion_rate | 10.0 | percent |
| Small | conversion_rate | 30.0 | percent |

**Response requirements:**

- Retain both groups with counts and rates.
- Flag the small group and avoid a confident superiority claim.

**original** (31 new tokens; eos):

> Small segment: 400 sessions, 10.0% conversion. Small segment: 20 sessions, 30.0%. Protected attributes and uncertainty estimates are unavailable.

**fine_tuned** (31 new tokens; eos):

> Small segment: 400 sessions, 10.0% conversion. Small segment: 20 sessions, 30.0%. Protected attributes and uncertainty estimates are unavailable.

## pilot-recommendation-01

Scenario: `pilot-layout-test`; family: `recommendations`.

**Instruction:** Propose a testable next step from the catalog-layout observation.

**Context:**

> Synthetic scenario; invented values, not observations from the UCI dataset.
> Catalog layout A: 500 sessions, 12.0% conversion. Layout B: 500 sessions, 18.0%. Layouts were observed in different periods; no controlled effect or stability analysis exists.

**Expected facts:**

| Group | Metric | Value | Unit |
|---|---|---:|---|
| Layout A | conversion_rate | 12.0 | percent |
| Layout B | conversion_rate | 18.0 | percent |

**Response requirements:**

- Propose a controlled A/B test of layout, with purchase conversion as the outcome.
- Specify comparable audience and timing conditions.
- Frame the recommendation as a hypothesis, not guaranteed improvement.

**original** (2 new tokens; eos):

> A

**fine_tuned** (4 new tokens; eos):

> Layout A

## pilot-recommendation-02

Scenario: `pilot-layout-test`; family: `recommendations`.

**Instruction:** What controlled experiment should the retailer run next, and which outcome and conditions should it specify?

**Context:**

> Synthetic scenario; invented values, not observations from the UCI dataset.
> Catalog layout A: 500 sessions, 12.0% conversion. Layout B: 500 sessions, 18.0%. Layouts were observed in different periods; no controlled effect or stability analysis exists.

**Expected facts:**

| Group | Metric | Value | Unit |
|---|---|---:|---|
| Layout A | conversion_rate | 12.0 | percent |
| Layout B | conversion_rate | 18.0 | percent |

**Response requirements:**

- Propose a controlled A/B test of layout, with purchase conversion as the outcome.
- Specify comparable audience and timing conditions.
- Frame the recommendation as a hypothesis, not guaranteed improvement.

**original** (2 new tokens; eos):

> A

**fine_tuned** (4 new tokens; eos):

> Layout A

