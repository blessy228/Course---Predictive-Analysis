"""
MDI3003 - Advanced Predictive Analytics
Experiment 07: Recommendation System from Customer Transaction Data (Random Forest)

Core 3-hour pipeline: load -> clean -> temporal split -> popularity baseline ->
candidate generation -> negative sampling -> feature engineering -> Random Forest
-> Top-K ranking -> evaluation -> error/cold-start analysis -> visualizations.

Dataset note:
loads the real UCI Online Retail dataset from
DATA_PATH (data/online_retail.xlsx: 541,909 rows, 01-Dec-2010 to 09-Dec-2011).
Set USE_SYNTHETIC = True to fall back to a synthetic generator with the same
schema (InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice,
CustomerID, Country) if the file isn't available.
"""

import os
import json
import platform
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import joblib

# 0. Config and reproducibility
SEED = 42
np.random.seed(SEED)
rng = np.random.default_rng(SEED)

USE_SYNTHETIC = False
DATA_PATH = "C:/Users/Sarah Blessy/OneDrive/Desktop/FallSemester/C2 - Advanced Predictive Analytics/lab-DA7/online+retail/Online Retail.xlsx"
OUT_DIR = "outputs"
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "artifacts"), exist_ok=True)

K_LIST = [5, 10, 20]
MAIN_K = 10
N_NEG_PER_USER = 50
CANDIDATE_MIN_BUYERS = 5     
CANDIDATE_CAP = 2000         

sns.set_style("whitegrid")


# 1. Load data
def make_synthetic_transactions(n_customers=400, n_items=250, n_transactions=20000):
    """Generates a UCI-Online-Retail-shaped transaction log with a realistic
    long-tail popularity structure and mild customer-item affinity, so the
    downstream pipeline has real signal to learn from."""
    customer_ids = [f"C{1000+i}" for i in range(n_customers)]
    item_ids = [f"S{2000+i}" for i in range(n_items)]
    countries = ["United Kingdom", "Germany", "France", "Ireland", "Spain"]

    item_weights = 1.0 / np.arange(1, n_items + 1)
    item_weights = item_weights / item_weights.sum()
    rng.shuffle(item_weights)

    n_pref = 8
    customer_prefs = {
        c: rng.choice(n_items, size=n_pref, replace=False) for c in customer_ids
    }

    start = pd.Timestamp("2010-12-01")
    end = pd.Timestamp("2011-12-09")
    span_days = (end - start).days

    rows = []
    invoice_counter = 536365
    for _ in range(n_transactions):
        cust = rng.choice(customer_ids)
        day_offset = rng.integers(0, span_days)
        ts = start + pd.Timedelta(days=int(day_offset), hours=int(rng.integers(8, 19)))

        # 70% chance the basket draws from the customer's preferred items
        if rng.random() < 0.7:
            item_idx = rng.choice(customer_prefs[cust])
        else:
            item_idx = rng.choice(n_items, p=item_weights)
        item = item_ids[item_idx]

        qty = int(rng.integers(1, 12))
        price = round(float(rng.uniform(0.5, 40.0)), 2)
        is_cancel = rng.random() < 0.02
        invoice_no = f"{'C' if is_cancel else ''}{invoice_counter}"
        invoice_counter += 1

        rows.append({
            "InvoiceNo": invoice_no,
            "StockCode": item,
            "Description": f"ITEM {item}",
            "Quantity": -qty if is_cancel else qty,
            "InvoiceDate": ts,
            "UnitPrice": price,
            "CustomerID": cust,
            "Country": rng.choice(countries, p=[0.85, 0.05, 0.04, 0.03, 0.03]),
        })

    df = pd.DataFrame(rows)
    drop_idx = rng.choice(df.index, size=int(0.03 * len(df)), replace=False)
    df.loc[drop_idx, "CustomerID"] = np.nan
    return df


def load_data():
    if USE_SYNTHETIC or not os.path.exists(DATA_PATH):
        if not USE_SYNTHETIC:
            print(f"'{DATA_PATH}' not found -> falling back to synthetic data.")
        df = make_synthetic_transactions()
    elif DATA_PATH.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(DATA_PATH)
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    else:
        df = pd.read_csv(DATA_PATH, encoding="ISO-8859-1")
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    required = {"InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID"}
    missing = required - set(df.columns)
    assert not missing, f"Missing columns: {missing}"
    return df


raw_df = load_data()
raw_rows = len(raw_df)

# 2. Cleaning (Section 10.3 / 6.1 integrity rules)
n_missing_id = raw_df["CustomerID"].isna().sum()

df = raw_df.dropna(subset=["CustomerID"]).copy()
df["CustomerID"] = df["CustomerID"].astype(int).astype(str)
df["is_cancel"] = df["InvoiceNo"].astype(str).str.startswith("C")
df = df[(~df["is_cancel"]) & (df["Quantity"] > 0) & (df["UnitPrice"] >= 0)].copy()
df["Amount"] = df["Quantity"] * df["UnitPrice"]
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

filtered_rows = len(df)

dataset_card = {
    "source": "synthetic (UCI Online Retail schema)" if USE_SYNTHETIC else DATA_PATH,
    "raw_rows": int(raw_rows),
    "filtered_rows": int(filtered_rows),
    "missing_customer_id_removed": int(n_missing_id),
    "unique_customers": int(df["CustomerID"].nunique()),
    "unique_items": int(df["StockCode"].nunique()),
    "date_range": [str(df["InvoiceDate"].min()), str(df["InvoiceDate"].max())],
    "return_cancellation_policy": "rows with InvoiceNo starting 'C' or Quantity<=0 removed before feature/target construction",
}
print("Dataset card:", json.dumps(dataset_card, indent=2))

# 3. Chronological split
t1 = df["InvoiceDate"].quantile(0.70)
t2 = df["InvoiceDate"].quantile(0.85)

train_hist = df[df.InvoiceDate < t1].copy()
val_future = df[(df.InvoiceDate >= t1) & (df.InvoiceDate < t2)].copy()
test_future = df[df.InvoiceDate >= t2].copy()

print(f"Cutoffs: train_end={t1}, val_end={t2}")
print(f"train_hist={len(train_hist)}, val_future={len(val_future)}, test_future={len(test_future)}")

# 4. Popularity baseline (Section 13.1)
item_popularity = (
    train_hist.groupby("StockCode")["InvoiceNo"].nunique().sort_values(ascending=False)
)
popular_items = item_popularity.index.tolist()


def popularity_recommend(seen_items, k=10):
    return [i for i in popular_items if i not in seen_items][:k]


# 5. Candidate universe
buyer_counts = train_hist.groupby("StockCode")["CustomerID"].nunique()
eligible_items = buyer_counts[buyer_counts >= CANDIDATE_MIN_BUYERS].sort_values(ascending=False)
eligible_candidate_items = eligible_items.index.tolist()[:CANDIDATE_CAP]
print(f"Eligible candidate catalog size: {len(eligible_candidate_items)}")

val_positive_sets = val_future.groupby("CustomerID")["StockCode"].apply(set)
covered, total = 0, 0
for cust, items in val_positive_sets.items():
    total += len(items)
    covered += len(items & set(eligible_candidate_items))
candidate_recall = covered / total if total else np.nan
print(f"Candidate recall (validation window): {candidate_recall:.3f}")

# 6. Feature engineering 
def customer_features(hist, cutoff):
    g = hist.groupby("CustomerID")
    out = g.agg(
        cust_txns=("InvoiceNo", "nunique"),
        cust_items=("StockCode", "nunique"),
        cust_qty=("Quantity", "sum"),
        cust_spend=("Amount", "sum"),
        cust_last=("InvoiceDate", "max"),
    ).reset_index()
    out["cust_recency_days"] = (cutoff - out["cust_last"]).dt.days
    out["cust_avg_basket"] = out["cust_qty"] / out["cust_txns"].clip(lower=1)
    return out.drop(columns="cust_last")


def item_features_fn(hist, cutoff):
    g = hist.groupby("StockCode")
    out = g.agg(
        item_txns=("InvoiceNo", "nunique"),
        item_buyers=("CustomerID", "nunique"),
        item_qty=("Quantity", "sum"),
        item_avg_price=("UnitPrice", "mean"),
        item_last=("InvoiceDate", "max"),
    ).reset_index()
    out["item_recency_days"] = (cutoff - out["item_last"]).dt.days
    return out.drop(columns="item_last")


def pair_features_fn(hist):
    return (
        hist.groupby(["CustomerID", "StockCode"])
        .agg(
            pair_purchases=("InvoiceNo", "nunique"),
            pair_qty=("Quantity", "sum"),
            pair_spend=("Amount", "sum"),
        )
        .reset_index()
    )


FEATURE_COLS = [
    "cust_txns", "cust_items", "cust_qty", "cust_spend", "cust_recency_days", "cust_avg_basket",
    "item_txns", "item_buyers", "item_qty", "item_avg_price", "item_recency_days",
    "pair_purchases", "pair_qty", "pair_spend",
]


def build_examples(hist, future, cutoff, candidates, n_neg=N_NEG_PER_USER):
    """Builds leakage-safe positive/negative user-item examples.
    hist    -> transactions strictly BEFORE cutoff (used for features)
    future  -> transactions in the target window (used only for labels)
    """
    cust_f = customer_features(hist, cutoff)
    item_f = item_features_fn(hist, cutoff)
    pair_f = pair_features_fn(hist)

    active_customers = cust_f["CustomerID"].tolist()
    future_pos = future.groupby("CustomerID")["StockCode"].apply(set)

    rows = []
    for cust in active_customers:
        seen_items = set(hist.loc[hist.CustomerID == cust, "StockCode"])
        positives = future_pos.get(cust, set()) & set(candidates)
        for item in positives:
            rows.append((cust, item, 1))
        neg_pool = list(set(candidates) - seen_items - positives)
        if neg_pool:
            n = min(n_neg, len(neg_pool))
            neg_items = rng.choice(neg_pool, size=n, replace=False)
            for item in neg_items:
                rows.append((cust, item, 0))

    ex = pd.DataFrame(rows, columns=["CustomerID", "StockCode", "label"])
    ex = ex.merge(cust_f, on="CustomerID", how="left")
    ex = ex.merge(item_f, on="StockCode", how="left")
    ex = ex.merge(pair_f, on=["CustomerID", "StockCode"], how="left")
    for c in ["pair_purchases", "pair_qty", "pair_spend"]:
        ex[c] = ex[c].fillna(0)
    ex[FEATURE_COLS] = ex[FEATURE_COLS].fillna(0)
    return ex


train_ex = build_examples(train_hist, val_future, t1, eligible_candidate_items)
X_train, y_train = train_ex[FEATURE_COLS], train_ex["label"]

val_ex = train_ex.copy()

test_hist = pd.concat([train_hist, val_future], ignore_index=True)
test_ex = build_examples(test_hist, test_future, t2, eligible_candidate_items)
X_test, y_test = test_ex[FEATURE_COLS], test_ex["label"]

print(f"Train examples: {len(train_ex)} (positives={y_train.sum()})")
print(f"Test examples:  {len(test_ex)} (positives={y_test.sum()})")

# 7. Train Random Forest
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=SEED,
    n_jobs=-1,
)
rf.fit(X_train, y_train)

test_scores = rf.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, test_scores) if y_test.nunique() > 1 else np.nan
pr_auc = average_precision_score(y_test, test_scores) if y_test.nunique() > 1 else np.nan
print(f"Test ROC-AUC: {roc_auc:.3f} | Test PR-AUC: {pr_auc:.3f}")

joblib.dump(rf, os.path.join(OUT_DIR, "models", "random_forest.joblib"))

# 8. Top-K recommendation generation (Section 14)
test_ex = test_ex.copy()
test_ex["score"] = test_scores


def recommend_for_user(user_id, scored_df, k=10):
    u = scored_df[scored_df.CustomerID == user_id]
    return u.sort_values("score", ascending=False)["StockCode"].head(k).tolist()


test_future_pos = test_future.groupby("CustomerID")["StockCode"].apply(set)
test_customers = [c for c in test_ex["CustomerID"].unique() if c in test_future_pos.index]

rf_recs = {c: recommend_for_user(c, test_ex, k=max(K_LIST)) for c in test_customers}
seen_by_cust = test_hist.groupby("CustomerID")["StockCode"].apply(set)
pop_recs = {c: popularity_recommend(seen_by_cust.get(c, set()), k=max(K_LIST)) for c in test_customers}

# 9. Evaluation metrics (Section 15)
def precision_at_k(recs, relevant, k):
    recs = recs[:k]
    return len(set(recs) & set(relevant)) / max(k, 1)


def recall_at_k(recs, relevant, k):
    if not relevant:
        return np.nan
    return len(set(recs[:k]) & set(relevant)) / len(set(relevant))


def hit_rate_at_k(recs, relevant, k):
    return float(len(set(recs[:k]) & set(relevant)) > 0)


def evaluate(recs_dict, k):
    p, r, h = [], [], []
    for c in test_customers:
        relevant = test_future_pos.get(c, set())
        recs = recs_dict.get(c, [])
        p.append(precision_at_k(recs, relevant, k))
        r.append(recall_at_k(recs, relevant, k))
        h.append(hit_rate_at_k(recs, relevant, k))
    return np.nanmean(p), np.nanmean(r), np.nanmean(h)


results = []
for k in K_LIST:
    pp, pr, ph = evaluate(pop_recs, k)
    rp, rr, rh = evaluate(rf_recs, k)
    results.append({"model": "Popularity", "K": k, "Precision@K": pp, "Recall@K": pr, "HitRate@K": ph})
    results.append({"model": "RandomForest", "K": k, "Precision@K": rp, "Recall@K": rr, "HitRate@K": rh})

results_df = pd.DataFrame(results)
print(results_df.round(4).to_string(index=False))
results_df.to_csv(os.path.join(OUT_DIR, "ranking_metrics.csv"), index=False)

# 10. Error / cold-start case audit (Section 16.1)
cases = []
for c in test_customers[:200]:
    relevant = test_future_pos.get(c, set())
    if not relevant:
        continue
    recs10 = rf_recs[c][:10]
    hits = set(recs10) & relevant
    hist_size = len(seen_by_cust.get(c, set()))
    cases.append({
        "CustomerID": c, "history_size": hist_size, "true_future_items": len(relevant),
        "top10_recs": recs10, "hits": len(hits),
    })
cases_df = pd.DataFrame(cases)
cases_df.to_csv(os.path.join(OUT_DIR, "error_analysis.csv"), index=False)

# 11. Visualizations 
def show(fig_name):
    """Save the current figure, then display it inline, then close it."""
    path = os.path.join(FIG_DIR, fig_name)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.show()
    plt.close()
    return path


saved_figs = []

# 1. Transaction volume over time
plt.figure(figsize=(9, 4))
df.set_index("InvoiceDate").resample("W")["InvoiceNo"].nunique().plot()
plt.title("Weekly Transaction Volume")
plt.xlabel("Week")
plt.ylabel("Distinct Invoices")
saved_figs.append(show("01_transaction_volume.png"))

# 2. Top 15 items by transaction count
plt.figure(figsize=(9, 5))
item_popularity.head(15).plot(kind="bar")
plt.title("Top 15 Items by Transaction Count")
plt.xlabel("StockCode")
plt.ylabel("Distinct Invoices")
saved_figs.append(show("02_top15_items.png"))

# 3. Customer purchase-frequency distribution
plt.figure(figsize=(8, 5))
cust_txn_counts = df.groupby("CustomerID")["InvoiceNo"].nunique()
sns.histplot(cust_txn_counts, bins=40)
plt.title("Customer Purchase-Frequency Distribution")
plt.xlabel("Number of Invoices per Customer")
plt.ylabel("Number of Customers")
saved_figs.append(show("03_customer_frequency_dist.png"))

# 4. Customer RFM distributions
cust_all = customer_features(df, df["InvoiceDate"].max() + pd.Timedelta(days=1))
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
sns.histplot(cust_all["cust_recency_days"], bins=30, ax=axes[0])
axes[0].set_title("Recency (days)")
sns.histplot(cust_all["cust_txns"], bins=30, ax=axes[1])
axes[1].set_title("Frequency (invoices)")
sns.histplot(cust_all["cust_spend"], bins=30, ax=axes[2])
axes[2].set_title("Monetary (spend)")
plt.suptitle("Customer RFM Distributions")
saved_figs.append(show("04_rfm_distributions.png"))

# 5. Class balance after negative sampling
plt.figure(figsize=(5, 4))
train_ex["label"].value_counts().sort_index().plot(kind="bar")
plt.xticks([0, 1], ["Negative (0)", "Positive (1)"], rotation=0)
plt.title("Class Balance After Negative Sampling (Train)")
plt.ylabel("Count")
saved_figs.append(show("05_class_balance.png"))

# 6. Feature-importance chart
importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values()
plt.figure(figsize=(8, 6))
importances.plot(kind="barh")
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
saved_figs.append(show("06_feature_importance.png"))

# 7. Precision@K and Recall@K vs K
plt.figure(figsize=(8, 5))
for model in ["Popularity", "RandomForest"]:
    sub = results_df[results_df.model == model]
    plt.plot(sub["K"], sub["Precision@K"], marker="o", label=f"{model} Precision@K")
    plt.plot(sub["K"], sub["Recall@K"], marker="s", linestyle="--", label=f"{model} Recall@K")
plt.title("Precision@K and Recall@K vs K")
plt.xlabel("K")
plt.ylabel("Score")
plt.legend()
saved_figs.append(show("07_precision_recall_vs_k.png"))

# 8. Popularity vs Random Forest ranking metrics (bar comparison at MAIN_K)
plt.figure(figsize=(7, 5))
sub = results_df[results_df.K == MAIN_K].set_index("model")[["Precision@K", "Recall@K", "HitRate@K"]]
sub.plot(kind="bar", ax=plt.gca())
plt.title(f"Popularity vs Random Forest (K={MAIN_K})")
plt.xticks(rotation=0)
plt.ylabel("Score")
saved_figs.append(show("08_popularity_vs_rf.png"))

# 9. Recommendation-score distribution for positives and negatives
plt.figure(figsize=(8, 5))
sns.histplot(data=test_ex, x="score", hue="label", bins=40, stat="density", common_norm=False)
plt.title("Recommendation-Score Distribution (Positives vs Negatives)")
plt.xlabel("Predicted Purchase Probability")
saved_figs.append(show("09_score_distribution.png"))

# 10. Catalog-coverage plot: how many distinct items appear in RF Top-10 vs popularity Top-10
rf_covered = set()
for recs in rf_recs.values():
    rf_covered.update(recs[:10])
pop_covered = set()
for recs in pop_recs.values():
    pop_covered.update(recs[:10])
plt.figure(figsize=(6, 5))
plt.bar(["Popularity", "Random Forest"], [len(pop_covered), len(rf_covered)])
plt.title("Catalog Coverage: Distinct Items Recommended (Top-10)")
plt.ylabel("Distinct Items Recommended")
saved_figs.append(show("10_catalog_coverage.png"))

# 12. Save artifacts for reproducibility (Section 26 checklist)
with open(os.path.join(OUT_DIR, "artifacts", "feature_schema.json"), "w") as f:
    json.dump({"feature_cols": FEATURE_COLS}, f, indent=2)

with open(os.path.join(OUT_DIR, "artifacts", "split_manifest.json"), "w") as f:
    json.dump({"train_end": str(t1), "val_end": str(t2), "seed": SEED}, f, indent=2)

with open(os.path.join(OUT_DIR, "artifacts", "candidate_policy.json"), "w") as f:
    json.dump({
        "min_buyers": CANDIDATE_MIN_BUYERS,
        "candidate_cap": CANDIDATE_CAP,
        "eligible_catalog_size": len(eligible_candidate_items),
        "candidate_recall_validation": candidate_recall,
    }, f, indent=2)

with open(os.path.join(OUT_DIR, "dataset_card.json"), "w") as f:
    json.dump(dataset_card, f, indent=2)

# reload check
reloaded = joblib.load(os.path.join(OUT_DIR, "models", "random_forest.joblib"))
assert np.allclose(reloaded.predict_proba(X_test.head(20))[:, 1], rf.predict_proba(X_test.head(20))[:, 1])
print("Model reload check passed.")

print("\nDone. Figures saved to:", FIG_DIR)
for p in saved_figs:
    print(" -", p)