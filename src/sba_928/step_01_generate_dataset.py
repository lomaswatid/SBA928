"""Prepare duplicate-aware source splits and validated supervised records in memory."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import re
from decimal import Decimal
from difflib import SequenceMatcher

import pandas as pd

from .common import DATA_PATH, ROOT, load_data, pct

LOGGER = logging.getLogger(__name__)
SEED = 42
ALGORITHM = "duplicate_group_stratified_greedy_v1"
SPLITS = ("train", "validation", "test")
RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}
TASK_ALLOCATION = {
    "train": (1, 3, 4, 5, 9, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24),
    "validation": (2, 6, 8),
    "test": (7, 10, 13),
}
TASK_KEYS = (
    "overall-conversion", "bounce-comparison", "exit-comparison", "product-engagement",
    "page-values", "visitor-comparison", "weekend-comparison", "monthly-extrema",
    "traffic-extrema", "region-extrema", "browser-extrema", "os-extrema",
    "special-day", "marketing-hypotheses", "competitive-capabilities", "outcome-fairness",
    "demographic-limits", "causality", "small-groups", "demographic-grounding",
    "traffic-limits", "region-limits", "executive-engagement", "follow-up",
)


def row_stats(df, group_col):
    """Calculate session counts and conversion for a split-local grouping."""
    return df.groupby(group_col, dropna=False)["Revenue"].agg(
        sessions="count", conversion="mean"
    ).reset_index()


def validate_source(df: pd.DataFrame) -> None:
    """Reject malformed source data before assigning rows or calculating facts."""
    required = {"Revenue", "Weekend", "VisitorType", "Month", "TrafficType", "Region",
                "Browser", "OperatingSystems", "SpecialDay", "BounceRates", "ExitRates",
                "PageValues", "ProductRelated", "ProductRelated_Duration"}
    if required - set(df.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(df.columns))}")
    if df.empty or df.isna().any().any():
        raise ValueError("Source must be nonempty and have no missing values")
    for name in ["Revenue", "Weekend"]:
        if not pd.api.types.is_bool_dtype(df[name]):
            raise ValueError(f"{name} must contain booleans")
    for name in df.select_dtypes(include="number").columns:
        if not all(math.isfinite(float(v)) for v in df[name]):
            raise ValueError(f"Non-finite source values: {name}")
    for name in ["BounceRates", "ExitRates", "SpecialDay"]:
        if not df[name].between(0, 1).all():
            raise ValueError(f"Invalid fractional values: {name}")
    for name in ["PageValues", "ProductRelated", "ProductRelated_Duration"]:
        if not (df[name] >= 0).all():
            raise ValueError(f"Negative values: {name}")
    for name in ["TrafficType", "Region", "Browser", "OperatingSystems"]:
        if not all(float(v).is_integer() and float(v) >= 1 for v in df[name]):
            raise ValueError(f"Invalid category codes: {name}")


def canonicalize_source_row(row: dict) -> str:
    """Normalize typed values in sorted column order; never use row position."""
    values = []
    for name in sorted(row):
        value = row[name]
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, bool):
            typed = ["bool", value]
        elif isinstance(value, (int, float)):
            number = Decimal(str(value))
            if not number.is_finite():
                raise ValueError("Non-finite canonical source value")
            typed = ["number", format(number.normalize(), "f") if number else "0"]
        elif isinstance(value, str):
            typed = ["string", value]
        else:
            raise TypeError(f"Unsupported source value for {name}")
        values.append([name, typed])
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def group_source_rows(df: pd.DataFrame) -> dict[str, list[int]]:
    """Keep exact duplicate canonical rows in one SHA-256 provenance group."""
    groups = {}
    canonical = {}
    for position, row in enumerate(df.to_dict(orient="records")):
        text = canonicalize_source_row(row)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in canonical and canonical[digest] != text:
            raise ValueError("Source SHA-256 collision")
        canonical[digest] = text
        groups.setdefault(digest, []).append(position)
    return dict(sorted(groups.items()))


def split_source_groups(df, groups: dict) -> dict[str, list[str]]:
    """Seeded stratified greedy assignment minimizes row-count deviations."""
    assigned = {name: [] for name in SPLITS}
    rng = random.Random(SEED)
    for outcome in [False, True]:
        candidates = [key for key in sorted(groups)
                      if bool(df.iloc[groups[key][0]]["Revenue"]) == outcome]
        rng.shuffle(candidates)
        # Stable size sort preserves seeded ordering for equal-size groups.
        candidates.sort(key=lambda key: -len(groups[key]))
        total = sum(len(groups[key]) for key in candidates)
        target = {name: total * RATIOS[name] for name in SPLITS}
        counts = dict.fromkeys(SPLITS, 0)
        for key in candidates:
            size = len(groups[key])
            destination = min(SPLITS, key=lambda name: (
                sum((counts[s] + (size if s == name else 0) - target[s]) ** 2
                    for s in SPLITS), SPLITS.index(name)))
            assigned[destination].append(key)
            counts[destination] += size
    return {name: sorted(keys) for name, keys in assigned.items()}


def select_extrema(grouped, minimum_sessions=30):
    """Require eligible groups; choose tied extrema by sample count then group label."""
    eligible = grouped[grouped["sessions"] >= minimum_sessions].copy()
    if len(eligible) < 2:
        raise ValueError(f"Need at least two eligible groups at threshold {minimum_sessions}")
    eligible["_tie_key"] = eligible.iloc[:, 0].astype(str)
    best = eligible.sort_values(["conversion", "sessions", "_tie_key"], ascending=[False, False, True]).iloc[0]
    worst = eligible.sort_values(["conversion", "sessions", "_tie_key"], ascending=[True, False, True]).iloc[0]
    return best, worst


def _task_template(instruction, target, category):
    """Materialize an existing task idea from split-local calculations."""
    return {"instruction": instruction, "target": target, "category": category}


def calculate_task_templates(df):
    """Construct all 24 targets using only the supplied source partition."""
    overall = df["Revenue"].mean()
    n_total = len(df)
    n_buy = int(df["Revenue"].sum())
    n_no = n_total - n_buy

    by_revenue = df.groupby("Revenue")[
        ["BounceRates", "ExitRates", "PageValues",
         "ProductRelated", "ProductRelated_Duration"]
    ].mean()

    purchaser = by_revenue.loc[True]
    non = by_revenue.loc[False]

    visitor = row_stats(df, "VisitorType")
    weekend = row_stats(df, "Weekend")
    month = row_stats(df, "Month")
    traffic = row_stats(df, "TrafficType")
    region = row_stats(df, "Region")
    browser = row_stats(df, "Browser")
    os_group = row_stats(df, "OperatingSystems")

    month_best, month_worst = select_extrema(month, minimum_sessions=50)
    traffic_best, traffic_worst = select_extrema(traffic, minimum_sessions=100)
    region_best, region_worst = select_extrema(region, minimum_sessions=100)
    browser_best, browser_worst = select_extrema(browser, minimum_sessions=100)
    os_best, os_worst = select_extrema(os_group, minimum_sessions=100)

    near_special = df.assign(NearSpecialDay=df["SpecialDay"] > 0)
    special = row_stats(near_special, "NearSpecialDay")
    special_map = {
        bool(r["NearSpecialDay"]): r for _, r in special.iterrows()
    }

    visitor_lines = []
    for _, r in visitor.sort_values("sessions", ascending=False).iterrows():
        visitor_lines.append(
            f"{r['VisitorType']}: {int(r['sessions']):,} sessions, "
            f"{pct(r['conversion'])} conversion"
        )

    weekend_map = {bool(r["Weekend"]): r for _, r in weekend.iterrows()}

    pairs = [
        _task_template(
            "You are a market research analyst. Summarize the overall purchase conversion in the dataset and identify one important modeling concern.",
            f"The dataset contains {n_total:,} sessions. {n_buy:,} ended in a purchase and "
            f"{n_no:,} did not, producing an observed conversion rate of {pct(overall)}. "
            f"The outcome is imbalanced because non-purchasing sessions are the clear majority. "
            "Any model evaluation should therefore avoid relying on accuracy alone and should "
            "inspect performance for the smaller purchasing class.",
            "Conversion"
        ),
        _task_template(
            "Compare the average bounce rates of purchasing and non-purchasing sessions. Do not make a causal claim.",
            f"Purchasing sessions have an average BounceRates value of "
            f"{purchaser['BounceRates']:.4f}, compared with {non['BounceRates']:.4f} for "
            "non-purchasing sessions. This is an observed association in the dataset; it does "
            "not establish that bounce rate itself causes the purchase outcome.",
            "Engagement"
        ),
        _task_template(
            "Compare average exit rates for purchasing and non-purchasing sessions and explain the market research implication.",
            f"Average ExitRates is {purchaser['ExitRates']:.4f} for purchasing sessions and "
            f"{non['ExitRates']:.4f} for non-purchasing sessions. The difference suggests that "
            "exit behavior is useful to investigate when studying conversion, but a controlled "
            "analysis would be needed before treating it as a causal driver.",
            "Engagement"
        ),
        _task_template(
            "Analyze product-related page engagement for sessions that purchased compared with sessions that did not.",
            f"Purchasing sessions average {purchaser['ProductRelated']:.2f} product-related pages "
            f"and {purchaser['ProductRelated_Duration']:.2f} units of product-page duration. "
            f"Non-purchasing sessions average {non['ProductRelated']:.2f} product-related pages "
            f"and {non['ProductRelated_Duration']:.2f} duration. These descriptive differences "
            "can help identify engagement patterns associated with conversion.",
            "Consumer behavior"
        ),
        _task_template(
            "Compare PageValues for purchasing and non-purchasing sessions without overstating what the metric proves.",
            f"Average PageValues is {purchaser['PageValues']:.2f} for purchasing sessions and "
            f"{non['PageValues']:.2f} for non-purchasing sessions. The observed difference is "
            "worth further analysis, but it should not be interpreted as proof that PageValues "
            "causes a purchase.",
            "Purchase intention"
        ),
        _task_template(
            "Compare new and returning visitor performance using only the supplied dataset.",
            "Observed visitor-type results are: " + "; ".join(visitor_lines) + ". "
            "These figures describe differences in this dataset. They do not show that visitor "
            "type alone caused the conversion difference.",
            "Visitor segmentation"
        ),
        _task_template(
            "Compare weekend and weekday conversion behavior and propose a test rather than a guaranteed strategy.",
            f"Weekend sessions total {int(weekend_map[True]['sessions']):,} with an observed "
            f"conversion rate of {pct(weekend_map[True]['conversion'])}. Non-weekend sessions "
            f"total {int(weekend_map[False]['sessions']):,} with a conversion rate of "
            f"{pct(weekend_map[False]['conversion'])}. A reasonable next step is an A/B test "
            "of campaign timing while holding offer and audience conditions as constant as practical.",
            "Timing"
        ),
        _task_template(
            "Identify the strongest and weakest observed conversion months, while avoiding unsupported causal claims.",
            f"Among months with sufficient observations for this comparison, "
            f"{month_best['Month']} has the highest observed conversion rate at "
            f"{pct(month_best['conversion'])} across {int(month_best['sessions']):,} sessions. "
            f"{month_worst['Month']} has the lowest at {pct(month_worst['conversion'])} across "
            f"{int(month_worst['sessions']):,} sessions. These are descriptive monthly "
            "differences and do not by themselves prove a seasonal cause.",
            "Market trends"
        ),
        _task_template(
            "Compare traffic types and identify the strongest and weakest observed conversion groups with meaningful sample sizes.",
            f"Using traffic types with at least 100 sessions, TrafficType "
            f"{traffic_best['TrafficType']} has the highest observed conversion rate "
            f"({pct(traffic_best['conversion'])}, {int(traffic_best['sessions']):,} sessions), "
            f"while TrafficType {traffic_worst['TrafficType']} has the lowest "
            f"({pct(traffic_worst['conversion'])}, {int(traffic_worst['sessions']):,} sessions). "
            "TrafficType is coded, so conclusions should stay at the code level unless external "
            "documentation supplies channel meanings.",
            "Traffic analysis"
        ),
        _task_template(
            "Analyze region-level conversion fairly without inventing geographic labels.",
            f"Among Region codes with at least 100 sessions, Region {region_best['Region']} has "
            f"the highest observed conversion rate ({pct(region_best['conversion'])}, "
            f"{int(region_best['sessions']):,} sessions) and Region {region_worst['Region']} has "
            f"the lowest ({pct(region_worst['conversion'])}, {int(region_worst['sessions']):,} sessions). "
            "The dataset uses coded regions, so the analysis should not assign real-world place "
            "names that are not supplied.",
            "Regional analysis"
        ),
        _task_template(
            "Compare browser categories while accounting for sample size and avoiding demographic inference.",
            f"Among browser groups with at least 100 sessions, Browser {browser_best['Browser']} "
            f"has the highest observed conversion rate at {pct(browser_best['conversion'])}, "
            f"while Browser {browser_worst['Browser']} has the lowest at "
            f"{pct(browser_worst['conversion'])}. Browser choice should not be used to infer "
            "unsupported demographic characteristics.",
            "Technology"
        ),
        _task_template(
            "Compare operating-system groups and give a cautious interpretation.",
            f"Among operating-system groups with at least 100 sessions, OperatingSystems "
            f"{os_best['OperatingSystems']} has the highest observed conversion rate "
            f"({pct(os_best['conversion'])}), while OperatingSystems {os_worst['OperatingSystems']} "
            f"has the lowest ({pct(os_worst['conversion'])}). This is descriptive and should "
            "lead to usability investigation rather than assumptions about the people using each system.",
            "Technology"
        ),
        _task_template(
            "Does being near a special day appear associated with conversion in this dataset?",
            f"Sessions with SpecialDay greater than zero have an observed conversion rate of "
            f"{pct(special_map[True]['conversion'])} across {int(special_map[True]['sessions']):,} sessions. "
            f"Sessions with SpecialDay equal to zero convert at {pct(special_map[False]['conversion'])} "
            f"across {int(special_map[False]['sessions']):,} sessions. This comparison is associative "
            "and does not isolate the effect of a special day from other factors.",
            "Market trends"
        ),
        _task_template(
            "Give three evidence-grounded areas an e-commerce company should investigate to improve conversion. Frame them as hypotheses.",
            f"Three testable areas are: (1) engagement and navigation, because purchasing and "
            f"non-purchasing sessions differ in average bounce and exit rates; (2) product-page "
            f"engagement, because purchasers average {purchaser['ProductRelated']:.2f} product-related "
            f"pages versus {non['ProductRelated']:.2f} for non-purchasers; and (3) visitor segmentation, "
            "because conversion varies across visitor types. Each should be treated as a hypothesis "
            "for further analysis or experimentation rather than a proven causal strategy.",
            "Marketing strategy"
        ),
        _task_template(
            "What competitive implications can be drawn from this dataset without pretending it contains competitor data?",
            "The dataset can support competitive implications about capabilities, not competitor rankings. "
            "An online retailer could investigate stronger conversion analytics, visitor segmentation, "
            "product-page experience, traffic-source measurement, and seasonal testing. The dataset "
            "contains no named competitor performance, so it cannot establish which competitor performs best.",
            "Competitive implications"
        ),
        _task_template(
            "Identify the most important fairness limitation created by the Revenue distribution.",
            f"Revenue is imbalanced: {n_buy:,} of {n_total:,} sessions ended in a purchase "
            f"({pct(overall)}), while {n_no:,} did not. If evaluation focuses only on overall "
            "accuracy or majority behavior, the purchasing class may be underrepresented. "
            "Evaluation should report class-specific measures and inspect both outcomes separately.",
            "Bias and fairness"
        ),
        _task_template(
            "What demographic conclusions can this dataset support?",
            "The dataset does not provide detailed attributes such as race, ethnicity, household "
            "income, education, or customer motivations. It therefore should not be used to make "
            "claims about those characteristics. Fair analysis should stay within the variables "
            "actually supplied and explicitly state these demographic limitations.",
            "Bias and fairness"
        ),
        _task_template(
            "Explain why correlation and causation must be separated when interpreting this shopping dataset.",
            "The dataset records observed session characteristics and purchase outcomes, but it is "
            "an aggregate descriptive analysis. A variable such as bounce rate, page value, month, or "
            "visitor type may be associated with Revenue without being the direct cause of Revenue. "
            "Market recommendations should therefore be framed as hypotheses for testing.",
            "Research methodology"
        ),
        _task_template(
            "Propose a fair way to compare market segments when some groups contain far fewer sessions than others.",
            "Report both conversion rate and session count for every segment, set a minimum sample "
            "threshold for headline comparisons, flag small groups, and avoid treating unstable "
            "small-sample rates as equivalent to estimates from large groups. Retain smaller groups "
            "for visibility rather than silently deleting them.",
            "Bias and fairness"
        ),
        _task_template(
            "How should a market research analyst prevent an AI model from inventing unsupported demographic explanations?",
            "Prompts should explicitly restrict the model to supplied variables, prohibit unsupported "
            "demographic inference, require evidence for each claim, and ask the model to label "
            "limitations. Human review should compare generated conclusions with the actual dataset "
            "before the findings are used for business decisions.",
            "Bias and fairness"
        ),
        _task_template(
            "Why should traffic-type findings be interpreted cautiously in this dataset?",
            "TrafficType is represented by coded categories. Conversion differences can be described "
            "at the code level, but the codes should not be renamed as specific marketing channels "
            "unless reliable documentation supplies that mapping. Sample size and confounding factors "
            "should also be considered before changing acquisition strategy.",
            "Traffic analysis"
        ),
        _task_template(
            "Why should region findings be interpreted cautiously in this dataset?",
            "Region is a coded variable. The analysis can compare Region codes by session count and "
            "conversion rate, but it should not invent city, state, or country labels. Regional "
            "differences are descriptive and may reflect other factors that are not controlled here.",
            "Regional analysis"
        ),
        _task_template(
            "Create a concise executive insight about purchase behavior using engagement metrics.",
            f"Purchasing sessions show a distinct engagement profile: average BounceRates is "
            f"{purchaser['BounceRates']:.4f} versus {non['BounceRates']:.4f} for non-purchasers, "
            f"and purchasers view {purchaser['ProductRelated']:.2f} product-related pages on average "
            f"versus {non['ProductRelated']:.2f}. These differences justify deeper conversion-funnel "
            "analysis, but they should not be presented as causal effects.",
            "Executive summary"
        ),
        _task_template(
            "Suggest a responsible next step after identifying a segment with a high observed conversion rate.",
            "First verify that the segment has a sufficient sample size and that the pattern is stable "
            "across time or validation subsets. Then formulate a controlled test or follow-up analysis. "
            "Do not automatically move budget or declare the segment superior based only on one "
            "descriptive conversion rate.",
            "Research methodology"
        ),
    ]

    return pairs


def build_contexts(df) -> dict[int, str]:
    """Render bounded facts and limitations without source provenance arrays."""
    means = df.groupby("Revenue")[["BounceRates", "ExitRates", "PageValues", "ProductRelated", "ProductRelated_Duration"]].mean()
    def engagement(columns):
        lines = ["Descriptive group means; purchase=false/true means no purchase/purchase."]
        if any(c in columns for c in ["BounceRates", "ExitRates"]):
            lines.append("Rates are fractions, not percentages.")
        if "ProductRelated_Duration" in columns:
            lines.append("Duration units unspecified.")
        lines.append("Purchase | " + " | ".join(columns))
        for outcome in [False, True]:
            lines.append(str(outcome) + " | " + " | ".join(f"{means.loc[outcome,c]:.4f}" if c in ["BounceRates", "ExitRates"] else f"{means.loc[outcome,c]:.2f}" for c in columns))
        return "\n".join(lines)
    def group(column, threshold=None):
        table = row_stats(df, column).sort_values(column)
        lines = ["Descriptive conversion rates; no causal effects established.",
                 f"{column} | Sessions | Purchase rate"]
        for _, row in table.iterrows():
            lines.append(f"{row[column]} | {int(row['sessions'])} | {pct(row['conversion'])}")
        if threshold:
            eligible = table[table['sessions'] >= threshold]
            high = eligible['conversion'].max(); low = eligible['conversion'].min()
            lines.append(f"Eligibility: at least {threshold} sessions. Tied extrema selected by descending count, then code/name.")
            lines.append("Highest-rate ties: " + ", ".join(str(v) for v in eligible.loc[eligible['conversion']==high,column]))
            lines.append("Lowest-rate ties: " + ", ".join(str(v) for v in eligible.loc[eligible['conversion']==low,column]))
        if column in ["TrafficType", "Region", "Browser", "OperatingSystems"]:
            lines.append("Coded groups; no channel/location/demographic mappings supplied.")
        return "\n".join(lines)
    total=len(df);buy=int(df['Revenue'].sum())
    outcome=f"Sessions: {total}; purchases: {buy}; non-purchases: {total-buy}; purchase rate: {pct(buy/total)}. Descriptive outcomes."
    metadata="Available fields: " + ", ".join(sorted(df.columns)) + ". No detailed race, ethnicity, income, education, motivation or competitor performance fields."
    contexts={1:outcome,2:engagement(['BounceRates']),3:engagement(['ExitRates']),
              4:engagement(['ProductRelated','ProductRelated_Duration']),5:engagement(['PageValues']),
              6:group('VisitorType'),7:group('Weekend'),8:group('Month',50),
              9:group('TrafficType',100),10:group('Region',100),11:group('Browser',100),
              12:group('OperatingSystems',100),16:outcome,17:metadata,20:metadata,
              18:"Available evidence consists of descriptive session characteristics and purchase outcomes. No causal identification or controlled-effect estimates are supplied.",
              19:group('VisitorType')+'\nGroups have unequal sample sizes; headline comparisons require a declared minimum sample size.',
              21:group('TrafficType'),22:group('Region'),
              23:engagement(['BounceRates','ProductRelated']),
              24:group('VisitorType')+'\nTemporal stability and controlled effects have not been evaluated.'}
    special=df.assign(NearSpecialDay=df['SpecialDay']>0)
    table=row_stats(special,'NearSpecialDay')
    contexts[13]='SpecialDay>0 defines near-special-day; false means SpecialDay=0. Descriptive statistics.\nNear special day | Sessions | Purchase rate\n'+'\n'.join(f"{row['NearSpecialDay']} | {int(row['sessions'])} | {pct(row['conversion'])}" for _,row in table.iterrows())
    contexts[14]=engagement(['BounceRates','ExitRates','ProductRelated'])+'\n'+group('VisitorType')
    contexts[15] = (
        engagement(['BounceRates', 'ExitRates', 'ProductRelated'])
        + "\nAvailable source dimensions: Revenue (purchase outcome), VisitorType, "
        "ProductRelated (product-page count), ProductRelated_Duration, TrafficType and Month. "
        "TrafficType codes have no verified channel-name mappings. "
        "No competitor-company performance or rankings are supplied. "
        "These are descriptive observations, not demonstrated causal effects or competitive advantages."
    )
    return contexts


def build_supervised_record(task, number, split, group_ids, source_hash, context):
    """Create a stable record; provenance is metadata, never model input."""
    digest=hashlib.sha256("\n".join(sorted(group_ids)).encode()).hexdigest()
    return {"record_id":f"FT-UCI468-v1-{split}-{TASK_KEYS[number-1]}-{digest[:16]}",
            **task, "context":context, "source_split":split,"evidence_group":f"{split}:{digest}",
            "evidence_digest":digest,"split":split,"task_id":TASK_KEYS[number-1],
            "source_group_ids":sorted(group_ids),"source_sha256":source_hash}


def construct_records(df, groups, assignments, source_hash):
    """Build the 24 exclusively allocated tasks from their source partitions."""
    records={};partitions={}
    for split in SPLITS:
        positions=sorted(i for key in assignments[split] for i in groups[key])
        partition=df.iloc[positions].copy();partitions[split]=partition
        if set(partition['Revenue'].unique()) != {False,True}:
            raise ValueError(f"Both Revenue outcomes required in {split}")
        tasks=calculate_task_templates(partition);contexts=build_contexts(partition)
        numbers = TASK_ALLOCATION[split]
        records[split]=sorted([build_supervised_record(tasks[n-1],n,split,assignments[split],source_hash,contexts[n]) for n in numbers],key=lambda r:r['record_id'])
    return records,partitions


def format_training_input(record):
    """Use only instruction and factual context, excluding provenance/references."""
    return record['instruction']+"\n\nContext:\n"+record['context']


def validate_records(records, partitions, groups, assignments):
    """Collect validation failures and collisions; never silently rewrite targets."""
    failures=[];collisions=[];near=[]
    required={'record_id','category','instruction','context','target','source_split','evidence_group','split'}
    all_records=[r for split in SPLITS for r in records[split]]
    allocated = [n for split in SPLITS for n in TASK_ALLOCATION[split]]
    if sorted(allocated) != list(range(1, 25)):
        failures.append('Each task must be assigned exactly once')
    for split in SPLITS:
        if len(records[split]) != len(TASK_ALLOCATION[split]) or {r['task_id'] for r in records[split]} != {TASK_KEYS[n-1] for n in TASK_ALLOCATION[split]}:
            failures.append(f'Incorrect task allocation: {split}')
    ids=[r['record_id'] for r in all_records]
    if len(ids)!=len(set(ids)):failures.append('Duplicate record IDs')
    all_groups=[key for split in SPLITS for key in assignments[split]]
    if len(all_groups)!=len(set(all_groups)) or set(all_groups)!=set(groups):
        failures.append('Source-group assignment not exclusive/exhaustive')
    for split in SPLITS:
        expected=calculate_task_templates(partitions[split]);contexts=build_contexts(partitions[split])
        for r in records[split]:
            if not required<=r.keys():failures.append(f"Missing required fields: {r.get('record_id')}");continue
            if any(not isinstance(r[k],str) or not r[k].strip() for k in required):failures.append(f"Empty/non-string field: {r['record_id']}")
            if r['split'] not in SPLITS or r['split']!=split or r['source_split']!=split:failures.append(f"Invalid split: {r['record_id']}")
            if r['source_group_ids']!=assignments[split]:failures.append(f"Foreign source evidence: {r['record_id']}")
            n=TASK_KEYS.index(r['task_id'])+1
            # Recompute the exact known template from this partition, checking group/metric attribution.
            if r['target']!=expected[n-1]['target'] or r['context']!=contexts[n]:failures.append(f"Grounding/template mismatch: {r['record_id']}")
            if n<=16 or n==23:
                target_numbers=re.findall(r'(?<![A-Za-z])\d[\d,]*(?:\.\d+)?%?',r['target'])
                context_numbers=set(re.findall(r'(?<![A-Za-z])\d[\d,]*(?:\.\d+)?%?',r['context']))
                # Numbered recommendations are structural labels rather than factual quantities.
                ignored={'1','2','3'} if n==14 else set()
                missing=[v for v in target_numbers if v.replace(',','') not in {x.replace(',','') for x in context_numbers} and v not in ignored]
                if missing:failures.append(f"Target numbers absent from context: {r['record_id']}: {missing}")
    def normalized(text):return ' '.join(text.split())
    for i,a in enumerate(all_records):
        for b in all_records[i+1:]:
            content_a=tuple(normalized(a[k]) for k in ['instruction','context','target'])
            content_b=tuple(normalized(b[k]) for k in ['instruction','context','target'])
            if content_a==content_b:collisions.append({'type':'exact_record','ids':[a['record_id'],b['record_id']]})
            if a['split']!=b['split']:
                if normalized(a['target'])==normalized(b['target']):collisions.append({'type':'exact_target_manual_review','ids':[a['record_id'],b['record_id']]})
                # Conservative near-duplicate screen includes evidence and target, not instructions alone.
                similarity=SequenceMatcher(None,normalized(a['context']+' '+a['target']),normalized(b['context']+' '+b['target']),autojunk=False).ratio()
                if similarity>=0.90:near.append({'ids':[a['record_id'],b['record_id']],'similarity':round(similarity,4)})
    if collisions:failures.append('Exact collisions require manual review')
    if near:failures.append('Near-duplicate candidates require manual review; similarity is not proof of leakage')
    return {'failures':failures,'exact_collisions':collisions,'near_duplicate_candidates':near,
            'record_ids_exclusive':len(ids)==len(set(ids)),
            'source_groups_exclusive':len(all_groups)==len(set(all_groups)),
            'passed':not failures}


def token_length_preflight(records):
    """Cached tokenizer only; measure untruncated source and target with special tokens."""
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('google/flan-t5-small',local_files_only=True)
    measurements=[]
    for split in SPLITS:
        for r in records[split]:
            source=len(tokenizer(format_training_input(r),truncation=False,add_special_tokens=True)['input_ids'])
            target=len(tokenizer(r['target'],truncation=False,add_special_tokens=True)['input_ids'])
            measurements.append({'record_id':r['record_id'],'split':split,'source_tokens':source,'target_tokens':target,'passed':source<=256 and target<=96})
    return measurements


def construct_manifest(df, groups, assignments, records, source_hash, validation, lengths):
    """Construct manifest data and prospective JSONL hashes without writing files."""
    files={}
    for split in SPLITS:
        payload=''.join(json.dumps(r,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n' for r in records[split]).encode('utf-8')
        files[f'data/processed/{split}.jsonl']=hashlib.sha256(payload).hexdigest()
    return {'schema_version':'supervised_uci468_v1','split_algorithm':ALGORITHM,'seed':SEED,
            'source_file':DATA_PATH.relative_to(ROOT).as_posix(),'source_sha256':source_hash,
            'source_row_counts':{s:sum(len(groups[k]) for k in assignments[s]) for s in SPLITS},
            'source_group_ids':assignments,'generated_record_counts':{s:len(records[s]) for s in SPLITS},
            'record_ids':{s:[r['record_id'] for r in records[s]] for s in SPLITS},
            'output_file_sha256':files,'hash_status':'prospective_in_memory_not_written',
            'leakage_checks':validation,'token_lengths':lengths,
            'test_policy':'Never use test results for hyperparameters or checkpoint selection.'}


def prepare_dataset_in_memory():
    """Prepare and audit the dataset without writing final artifacts."""
    source_hash=hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    df=load_data();validate_source(df)
    if hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()!=source_hash:raise ValueError('Source changed while reading')
    groups=group_source_rows(df);assignments=split_source_groups(df,groups)
    records,partitions=construct_records(df,groups,assignments,source_hash)
    validation=validate_records(records,partitions,groups,assignments)
    lengths=token_length_preflight(records)
    manifest=construct_manifest(df,groups,assignments,records,source_hash,validation,lengths)
    return df,groups,assignments,records,partitions,validation,lengths,manifest


def main() -> None:
    """Prepare and report dry-run checks; final artifact writing is not enabled."""
    logging.basicConfig(level=logging.INFO,format='%(levelname)s %(message)s')
    LOGGER.info('Starting dataset-grounded example generation and splitting')
    _,_,_,records,_,validation,lengths,_=prepare_dataset_in_memory()
    LOGGER.info('Record counts: %s',{s:len(records[s]) for s in SPLITS})
    LOGGER.info('Validation findings: %s',validation)
    LOGGER.info('Overlong records: %s',[r for r in lengths if not r['passed']])
    if not validation['passed'] or any(not r['passed'] for r in lengths):
        raise ValueError('Dry-run findings require review; no artifacts were written')


if __name__ == '__main__':
    main()
