# PharmGKB + Gemma 2B Pharmacogenomic Recommendation Engine

## What the uploaded data establishes

- CPIC gene-drug guideline pairs: 319
- PharmGKB gene-chemical relationship rows: 11035
- Unique gene-drug pairs in relationships.tsv: 11035
- Associated relationships: 6536
- Not-associated relationships: 2126
- Ambiguous relationships: 2373
- Drug-label records: 1433
- Expanded gene-drug pairs from labels: 2024
- CPIC pairs overlapping the PharmGKB gene-chemical relationship table: 314

## Important scientific constraint

The uploaded files are curated knowledge/annotation tables. They do NOT constitute a passage-level NLP corpus.
Do not claim that Gemma has been fine-tuned or that extraction precision/recall has been measured until real PharmGKB text passages with gold annotations are supplied.

## Recommended experiment

1. Build a passage-level dataset from PharmGKB evidence text.
2. Use pair-aware train/validation/test splitting to prevent the same gene-drug pair from appearing across partitions.
3. Fine-tune Gemma 2B with LoRA for structured JSON extraction.
4. Evaluate entity and relation extraction separately.
5. Convert accepted extractions into numerical evidence features.
6. Compare:
   - structured clinical/genetic baseline
   - Gemma-derived features alone
   - structured + Gemma fused model
7. Keep the clinical test set untouched until the final comparison.

## Files

- `gene_drug_gold_standard.csv`: curated PharmGKB gene-chemical relationships enriched with CPIC evidence.
- `drug_label_gene_drug_pairs.csv`: expanded drug-label gene-drug evidence.
- `pair_aware_split.csv`: leakage-resistant pair split.
- `pharmgkb_extraction_template.jsonl`: schema template only, not experimental data.
- `train_gemma_lora.py`: Gemma 2B LoRA training scaffold.
- `evaluate_extraction.py`: relation-level precision/recall/F1 evaluator.
