# Requirement 2 - Prompt Variations

Each research question is expressed at four levels. The progression adds specificity, analyst context, output constraints, and grounding rules.

## 1. Consumer behavior

**Basic:** Analyze customer purchasing behavior in the online shopping dataset.

**Specific:** Compare the browsing behavior of sessions that generated revenue with sessions that did not, focusing on product-related pages, duration, bounce rate, exit rate, and page value.

**Role + context:** You are a market research analyst for an e-commerce company. Compare purchasing and non-purchasing sessions using engagement and navigation variables, then identify two business implications.

**Grounded + constrained:** Using only the supplied dataset statistics, compare purchasing and non-purchasing sessions using product-related pages, product-related duration, bounce rate, exit rate, and page value. Report three findings. Do not infer causes that the dataset cannot establish.

## 2. Visitor segmentation

**Basic:** Compare new and returning visitors.

**Specific:** Compare conversion rates and product-page engagement for New_Visitor and Returning_Visitor sessions.

**Role + context:** Act as an e-commerce market research analyst. Compare new and returning visitors and explain how the observed differences could inform retention and acquisition strategy.

**Grounded + constrained:** Using only the provided visitor-type statistics, compare New_Visitor and Returning_Visitor conversion rates and engagement. Distinguish observed associations from causal explanations.

## 3. Market trends

**Basic:** Identify seasonal purchasing patterns.

**Specific:** Compare monthly session volume and conversion rate to identify high- and low-performing months.

**Role + context:** You are preparing a seasonal market research brief. Identify the strongest and weakest months by observed conversion rate and discuss possible marketing implications.

**Grounded + constrained:** Using only the supplied monthly statistics, identify the highest and lowest observed conversion months. Do not claim seasonality causes conversion differences without additional evidence.

## 4. Traffic analysis

**Basic:** Analyze traffic sources.

**Specific:** Compare conversion rates across traffic types and identify traffic sources with enough sessions to support a meaningful descriptive comparison.

**Role + context:** Act as a digital marketing analyst. Compare traffic-source performance and identify which sources deserve further investigation for acquisition strategy.

**Grounded + constrained:** Using only supplied traffic-type session counts and conversion rates, identify notable traffic-source differences. Flag small samples and avoid claiming that traffic source caused the observed conversion rate.

## 5. Engagement

**Basic:** Analyze bounce rate and purchasing.

**Specific:** Compare average BounceRates for purchasing and non-purchasing sessions.

**Role + context:** You are evaluating website engagement. Explain how bounce rate differs between sessions that generated revenue and those that did not, and give one testable business hypothesis.

**Grounded + constrained:** Use only the supplied average bounce-rate statistics. Describe the association between bounce rate and Revenue, but do not state that bounce rate causes purchases or non-purchases.

## 6. Engagement

**Basic:** Analyze exit rate and purchasing.

**Specific:** Compare average ExitRates for purchasing and non-purchasing sessions.

**Role + context:** Act as a conversion analyst. Compare exit behavior between purchasing and non-purchasing sessions and identify a follow-up analysis the company should perform.

**Grounded + constrained:** Using only the supplied ExitRates statistics, describe differences between revenue outcomes. Treat the result as descriptive, not causal.

## 7. Timing

**Basic:** Compare weekend and weekday shoppers.

**Specific:** Compare session counts and conversion rates between weekend and non-weekend sessions.

**Role + context:** You are planning campaign timing. Compare weekend and weekday conversion behavior and explain how the result could guide an A/B test.

**Grounded + constrained:** Using only the supplied weekend statistics, compare observed conversion rates. Do not recommend reallocating budget solely from this descriptive comparison.

## 8. Technology

**Basic:** Analyze browser purchasing patterns.

**Specific:** Compare browser categories by session count and conversion rate, while accounting for small sample sizes.

**Role + context:** Act as a digital experience analyst. Identify browser groups that may deserve usability testing based on session volume and observed conversion.

**Grounded + constrained:** Use only supplied browser counts and conversion rates. Flag small groups and do not infer user demographics from browser choice.

## 9. Regional analysis

**Basic:** Analyze regional purchasing patterns.

**Specific:** Compare available Region categories by session count and conversion rate.

**Role + context:** Act as a market research analyst. Compare the coded Region groups and identify where further localized research may be justified.

**Grounded + constrained:** Use only the supplied Region codes, session counts, and conversion rates. Do not convert region codes into geographic locations unless the dataset explicitly provides that mapping.

## 10. Purchase intention

**Basic:** Analyze page value and purchase intention.

**Specific:** Compare average PageValues between purchasing and non-purchasing sessions.

**Role + context:** Act as an e-commerce analyst. Explain how PageValues differs by Revenue outcome and why it may be useful for conversion analysis.

**Grounded + constrained:** Use only the supplied PageValues statistics. Describe association rather than claiming PageValues causes a purchase.

## 11. Marketing strategy

**Basic:** Suggest ways to improve conversion.

**Specific:** Based on observed engagement, visitor, timing, and traffic patterns, identify three areas that should be tested to improve conversion.

**Role + context:** You are a market research analyst advising an e-commerce manager. Turn the supplied descriptive findings into three testable marketing or UX experiments.

**Grounded + constrained:** Using only the supplied dataset findings, propose three experiments. Label each recommendation as a hypothesis to test, not a proven causal strategy.

## 12. Competitive implications

**Basic:** Identify competitive insights from the dataset.

**Specific:** Identify customer-experience capabilities an online retailer may need to strengthen based on the observed consumer behavior.

**Role + context:** Act as a market intelligence analyst. Translate the observed shopping-behavior patterns into capabilities an e-commerce retailer should investigate to remain competitive.

**Grounded + constrained:** Using only the supplied consumer-behavior findings, identify competitive implications. Do not rank or name competitors because the dataset contains no competitor-company performance data.
