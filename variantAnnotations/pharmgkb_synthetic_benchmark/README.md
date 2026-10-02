# PharmGKB Synthetic Pharmacogenomic Benchmark

This is a SYNTHETIC benchmark for development and methodology testing.

It is NOT a real patient cohort. The synthetic outcome labels were generated
from curated CPIC/PharmGKB relationships plus stochastic patient covariates.
Therefore the resulting accuracy must NOT be reported as clinical accuracy,
patient-level performance, or evidence of clinical effectiveness.

Use this benchmark to test:
1. data plumbing
2. feature engineering
3. model training
4. confusion-matrix/error-analysis code
5. the eventual Gemma feature-fusion architecture

The intended next step is to replace the synthetic outcome generator with a
real, appropriately licensed/available pharmacogenomic patient cohort.
