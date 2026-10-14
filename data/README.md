# Data provenance

The source CSV is the public UCI Online Shoppers Purchasing Intention Dataset (ID 468), credited to C. Sakar and Y. Kastro (2018), DOI https://doi.org/10.24432/C5F88Q, licensed CC BY 4.0. It was retrieved through ucimlrepo and exported with 17 predictors plus Revenue. The source is acknowledged here and in FINAL_REPORT.md.

`processed/manifest.json` binds the source and frozen train/validation/test JSONL files by SHA-256. The source partitions contain 9,864 / 1,233 / 1,233 sessions; supervised records contain 18 / 3 / 3 examples. Exact duplicate source rows stay within one split. These records contain public-source statistics and provenance hashes, not private customer identities.

`evaluation/dev_pilot_v1/` is explicitly hand-authored synthetic development evidence: 16 prompts in 15 scenarios, including one intentional paraphrase pair. It is not real market research data or a final benchmark.

The files are frozen. No dataset was regenerated during final submission preparation. Training accesses train/validation; the original held-out comparison requires explicit authorization. Existing tests inspect split metadata and synthetic/development fixtures without loading models.
