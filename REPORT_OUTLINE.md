# SBA 928 Report Outline

## 1. Introduction
Explain the purpose of the assessment and the project question.

## 2. Dataset Selection
Describe the UCI Online Shoppers Purchasing Intention Dataset:
- Source
- Number of sessions
- Main variables
- Revenue target
- Why the dataset is relevant to market research

## 3. Requirement 1 - Diverse Prompt Generation
Insert a table of prompts from `prompts/prompt_catalog.json`.
Explain how the prompts cover:
- Consumer behavior
- Conversion
- Market trends
- Segmentation
- Traffic sources
- Marketing/business strategy

## 4. Requirement 2 - Prompt Variations
Use `prompts/prompt_variations.md`.
Discuss:
- Broad/basic prompts
- Specific prompts
- Role/context prompting
- Grounded/constrained prompting
- Why greater specificity can improve usefulness and reduce unsupported claims

## 5. Requirement 3 - Fine-Tuning Experiment
Document:
- Model: google/flan-t5-small
- Why a small instruction-following model was selected
- How dataset statistics were converted into prompt/response examples
- Train/test split
- Number of epochs
- Batch size
- Learning rate
- Training/evaluation loss from your run
- Hardware used

Do not invent training results. Insert the values produced by your own run.

## 6. Requirement 4 - Original vs Fine-Tuned Model
Use `submission/evidence/model_comparison/model_comparison_evaluation.csv`.

Compare:
- Relevance
- Accuracy against the dataset
- Market-research terminology
- Specificity
- Actionability
- Unsupported claims
- Supplemental token-overlap score

State whether fine-tuning improved, worsened, or produced mixed results.

## 7. Requirement 5 - Bias Assessment
Use `submission/evidence/bias_audit/bias_report.md`.

Discuss:
- Revenue class imbalance
- Representation across groups
- Dataset age/time limitation
- Lack of detailed demographic variables
- Risk of overgeneralizing visitor behavior
- Risk of confusing correlation with causation
- Model bias inherited from the pretrained model

## 8. Requirement 6 - Fairness Strategies
Discuss:
- Analyze minority and majority classes separately
- Compare results across available segments
- Require dataset-grounded responses
- Prevent unsupported demographic inference
- Clearly state limitations
- Human review of AI-generated conclusions
- Repeat evaluation when new data becomes available

## 9. Conclusion
Summarize what prompt engineering and fine-tuning did and did not improve.

## 10. References
Include:
- UCI dataset citation/DOI
- FLAN-T5 model/model card
- Course material if required
