# MDI3003 Lab 04 - Customer Segment Prediction (Online Retail II, Research Extension)

## What this run does
Dataset C (UCI Online Retail II) has no predefined segment label, so per the lab manual's
Dataset-use rule, a frozen two-stage procedure was used:

1. Stage 1: an RFM-based rule (documented in artifacts/label_definition.md) assigns every
   customer to one of 4 segments: Champions, Loyal Customers, At Risk, Hibernating or Lost.
2. Stage 2: four core classifiers (Dummy, GaussianNB, BernoulliNB, CategoricalNB) are trained
   with leakage-safe, training-only-fit pipelines to PREDICT that segment from a separate,
   non-circular feature set (country, avg unit price, bulk-buy ratio, avg basket size,
   product diversity, weekend ratio, tenure, return rate). Raw R/F/M values are excluded from
   the predictors to avoid label circularity.

## Selected model
BernoulliNB, selected by mean 5-fold stratified cross-validation macro F1 on the
training split only, then evaluated once on the locked 20% test split.

## Key result files (results/)
- data_audit.csv               : missingness/uniqueness audit of the customer-level table
- class_distribution.csv       : segment counts
- cv_results.csv                : 5-fold CV comparison across all 4 core models
- classification_report.csv    : per-class precision/recall/F1 on the locked test set
- test_summary.csv              : accuracy, macro F1, weighted F1, timing on the locked test set
- confusion_matrix_normalized.csv
- test_predictions.csv          : RESULT COLUMN for the locked test customers
                                   (predicted_segment, max_posterior, confidence_category)
- full_customer_predictions.csv : RESULT COLUMN for every customer in the dataset (train+test)
- error_analysis.csv            : misclassified test customers with feature values

## Figures (figures/)
- class_distribution.png, cv_comparison.png, confusion_matrix.png

## Artifacts (artifacts/)
- label_definition.md   : frozen Stage-1 RFM label rule + circularity audit
- feature_manifest.json : demographic/psychographic/behavioral feature groups
- dataset_card.json     : dataset provenance, licence, privacy note
- split_manifest.csv    : which Customer ID went to train vs test
- versions.json          : package versions for reproducibility

## Models (models/)
- selected_pipeline.joblib : the fitted, deployable scikit-learn pipeline

## Limitations (read before using this for real decisions)
- The segment label is derived from the data itself (RFM rule), not an externally verified
  business segmentation, so it should be treated as an analytical construct, not ground truth.
- avg_unit_price and bulk_buy_ratio are behaviorally inferred psychographic proxies, not
  measured attitudes or survey data.
- Country is the only demographic-like field available in this transactional dataset.
- All data is UK-dominant (about 92% of rows), so segment definitions may not generalize well
  to other countries with very few customers.
