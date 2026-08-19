# Frozen Label-Construction Procedure (Stage 1)

Dataset: UCI Online Retail II. This dataset has NO predefined customer-segment label, so one
was constructed using a standard, documented RFM (Recency, Frequency, Monetary) scoring rule,
BEFORE any supervised model was trained. This rule was frozen and not modified after seeing
model results.

Snapshot date: max(InvoiceDate) + 1 day = 2011-12-10 12:50:00

For each customer (sales lines only; cancellations excluded):
  Recency  = days since the customer's most recent purchase
  Frequency = number of distinct invoices
  Monetary  = total amount spent (Quantity x Price, summed)

Each of R, F, M is scored 1 (worst) to 4 (best) using quartiles across all customers.
RFM_total = R_score + F_score + M_score  (range 3 to 12)

Segment mapping (frozen business rule):
  RFM_total 10-12  -> Champions
  RFM_total 7-9    -> Loyal Customers
  RFM_total 4-6    -> At Risk
  RFM_total 3      -> Hibernating or Lost

## Label-Provenance and Circularity Audit (manual section 7.5)
Recency, Frequency, and Monetary (and their R/F/M scores) directly define this label by
construction. They are therefore EXCLUDED from the Stage 2 predictor set, because including
them would let the classifier trivially reconstruct the label it was derived from
(label circularity). Stage 2 instead uses an independent behavioral/demographic/psychographic
proxy feature set (country, average unit price, bulk-buy ratio, average basket size, product
diversity, return rate, tenure, weekend purchase ratio) that is correlated with, but not
deterministic of, the RFM label.

## Psychographic Measurement Provenance (manual section 7.6)
This dataset contains no self-reported psychographic survey data. The "average unit price"
and "bulk-buy ratio" features used in Stage 2 are BEHAVIORALLY INFERRED proxies for price
sensitivity and deal-seeking attitude, not directly measured psychographic constructs. They
should be interpreted with caution and are documented here as inferred, not ground truth.
