# # MDI3003 Advanced Predictive Analytics
# ## Lab 05: Product and Brand Sentiment Prediction from Tweet Data
# 
# Student Name: Sarah Blessy
# Registration No: 23MID0296
# Core dataset: D2 Twitter US Airline Sentiment (Kaggle: crowdflower/twitter-airline-sentiment)

import os
import re
import json
import time
import platform
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_recall_fscore_support
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

SEED = 42
np.random.seed(SEED)
OUT = 'lab05_outputs'
os.makedirs(OUT, exist_ok=True)
print(platform.python_version())


# ## 1. Dataset configuration and loading

DATA_PATH = 'C:/FallSemester/C2 - Advanced Predictive Analytics/lab-DA5/archive/Tweets.csv'
TEXT_COL = 'text'
TARGET_COL = 'airline_sentiment'
ID_COL = 'tweet_id'
ENTITY_COL = 'airline'

raw = pd.read_csv(DATA_PATH)
required = {TEXT_COL, TARGET_COL}
missing = required - set(raw.columns)
assert not missing, missing

keep_cols = [c for c in [ID_COL, TEXT_COL, TARGET_COL, ENTITY_COL] if c in raw.columns]
df = raw[keep_cols].copy()
df = df.dropna(subset=[TEXT_COL, TARGET_COL])
df[TEXT_COL] = df[TEXT_COL].astype(str)
df.shape


# ## 2. Data audit

print(df.shape)
print(df.columns.tolist())
print(df[TARGET_COL].value_counts(dropna=False))
print(df.isna().sum())
print('Duplicate IDs', df[ID_COL].duplicated().sum() if ID_COL in df.columns else 'N/A')
print('Duplicate text', df[TEXT_COL].duplicated().sum())
df.head(5)


# ## 3. Leakage and identifier removal
# 
# The raw Kaggle file also contains sentiment_confidence, negativereason, name, tweet_coord and other identifier or annotation fields. These are excluded because they either leak the target label or carry personal identifiers that are not needed for text based sentiment classification. Only tweet_id, text, airline_sentiment and airline are retained above.


# ## 4. Minimal tweet normalization
# 
# URLs are replaced with a URL token and mentions are replaced with a USER token. Whitespace is collapsed. Negation, emojis, hashtags and punctuation are preserved because they carry sentiment signal.

def normalize_tweet(text):
    text = str(text)
    text = re.sub(r'https?://\S+|www\.\S+', ' URLTOKEN ', text)
    text = re.sub(r'@\w+', ' USERTOKEN ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df[TEXT_COL].map(normalize_tweet)
print('Duplicate clean text', df['clean_text'].duplicated().sum())
df[['clean_text', TARGET_COL]].head(5)


# ## 5. Exploratory visualizations
# 
# Class distribution and tweet length distribution, computed on the full audited dataset.

class_counts = df[TARGET_COL].value_counts()
plt.figure(figsize=(6, 4))
class_counts.plot(kind='bar', color=['#c0392b', '#7f8c8d', '#27ae60'])
plt.title('Sentiment class distribution')
plt.xlabel('Sentiment')
plt.ylabel('Tweet count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

df['tweet_length'] = df['clean_text'].str.split().map(len)
plt.figure(figsize=(6, 4))
plt.hist(df['tweet_length'], bins=30, color='#2980b9')
plt.title('Tweet length distribution (words)')
plt.xlabel('Word count')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


# ## 6. Fixed stratified split
# 
# A single stratified train/test split is created and frozen. All comparative models use this same split. Row manifests are saved so the split is reproducible.

train_df, test_df = train_test_split(
    df, test_size=0.20, random_state=SEED, stratify=df[TARGET_COL]
)
train_df.to_csv(os.path.join(OUT, 'train_manifest.csv'), index=False)
test_df.to_csv(os.path.join(OUT, 'test_manifest.csv'), index=False)

X_train, y_train = train_df['clean_text'], train_df[TARGET_COL]
X_test, y_test = test_df['clean_text'], test_df[TARGET_COL]
print(X_train.shape, X_test.shape)


# ## 7. Top terms by class (training data only)
# 
# Top TF-IDF weighted unigrams for each sentiment class, computed strictly from the training partition to avoid leakage.

top_vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=3, max_df=0.95, stop_words='english')
top_matrix = top_vectorizer.fit_transform(X_train)
feature_names = np.array(top_vectorizer.get_feature_names_out())

fig, axes = plt.subplots(1, len(class_counts.index), figsize=(5 * len(class_counts.index), 4))
if len(class_counts.index) == 1:
    axes = [axes]
for ax, cls in zip(axes, sorted(y_train.unique())):
    mask = (y_train.values == cls)
    mean_scores = np.asarray(top_matrix[mask].mean(axis=0)).ravel()
    top_idx = mean_scores.argsort()[-10:]
    ax.barh(feature_names[top_idx], mean_scores[top_idx], color='#8e44ad')
    ax.set_title('Top terms: ' + str(cls))
plt.tight_layout()
plt.show()


# ## 8. Baselines: Dummy and VADER
# 
# A trivial majority class baseline and a no training lexicon based baseline are established before any learned model is trained.

dummy = Pipeline([
    ('tfidf', TfidfVectorizer(min_df=2)),
    ('clf', DummyClassifier(strategy='most_frequent', random_state=SEED))
])
dummy.fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)
dummy_macro_f1 = f1_score(y_test, dummy_pred, average='macro')
print('Dummy macro F1', dummy_macro_f1)

analyzer = SentimentIntensityAnalyzer()

def vader_label(text):
    compound = analyzer.polarity_scores(text)['compound']
    if compound >= 0.05:
        return 'positive'
    if compound <= -0.05:
        return 'negative'
    return 'neutral'

vader_pred = X_test.map(vader_label)
vader_macro_f1 = f1_score(y_test, vader_pred, average='macro')
print('VADER macro F1', vader_macro_f1)


# ## 9. Classical models and cross validation
# 
# MultinomialNB, Logistic Regression and LinearSVC are each fitted inside a TF-IDF pipeline and compared under identical five fold stratified cross validation on the training partition only. The test set is not touched at this stage.

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

models = {
    'MultinomialNB': Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.98, sublinear_tf=True)),
        ('clf', MultinomialNB(alpha=0.5))
    ]),
    'LogisticRegression': Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.98, sublinear_tf=True)),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=SEED))
    ]),
    'LinearSVC': Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.98, sublinear_tf=True)),
        ('clf', LinearSVC(class_weight='balanced', random_state=SEED))
    ])
}

rows = []
for name, pipe in models.items():
    scores = cross_validate(
        pipe, X_train, y_train, cv=cv,
        scoring={'macro_f1': 'f1_macro', 'weighted_f1': 'f1_weighted', 'accuracy': 'accuracy'},
        n_jobs=-1, return_train_score=False
    )
    rows.append({
        'model': name,
        'macro_f1_mean': scores['test_macro_f1'].mean(),
        'macro_f1_sd': scores['test_macro_f1'].std(),
        'weighted_f1_mean': scores['test_weighted_f1'].mean(),
        'accuracy_mean': scores['test_accuracy'].mean(),
        'fit_time_mean': scores['fit_time'].mean()
    })

rows.append({
    'model': 'Dummy',
    'macro_f1_mean': dummy_macro_f1,
    'macro_f1_sd': np.nan,
    'weighted_f1_mean': np.nan,
    'accuracy_mean': accuracy_score(y_test, dummy_pred),
    'fit_time_mean': np.nan
})
rows.append({
    'model': 'VADER',
    'macro_f1_mean': vader_macro_f1,
    'macro_f1_sd': np.nan,
    'weighted_f1_mean': np.nan,
    'accuracy_mean': accuracy_score(y_test, vader_pred),
    'fit_time_mean': np.nan
})

cv_results = pd.DataFrame(rows).sort_values('macro_f1_mean', ascending=False).reset_index(drop=True)
cv_results.to_csv(os.path.join(OUT, 'cv_results.csv'), index=False)
cv_results

plot_results = cv_results[cv_results['model'].isin(['MultinomialNB', 'LogisticRegression', 'LinearSVC'])]
plt.figure(figsize=(6, 4))
plt.bar(plot_results['model'], plot_results['macro_f1_mean'], yerr=plot_results['macro_f1_sd'], color='#16a085', capsize=5)
plt.title('Cross validation macro F1 by model')
plt.ylabel('Macro F1')
plt.tight_layout()
plt.show()


# ## 10. Model selection and locked test evaluation
# 
# The model with the highest cross validation macro F1 among the three learned classifiers is selected. It is then evaluated exactly once on the locked test set.

learned_results = cv_results[cv_results['model'].isin(models.keys())].reset_index(drop=True)
best_name = learned_results.iloc[0]['model']
best_model = models[best_name]
best_model.fit(X_train, y_train)
test_pred = best_model.predict(X_test)
print('Selected model', best_name)
print(classification_report(y_test, test_pred, digits=4))

macro_f1_test = f1_score(y_test, test_pred, average='macro')
weighted_f1_test = f1_score(y_test, test_pred, average='weighted')
accuracy_test = accuracy_score(y_test, test_pred)
print('Macro F1', macro_f1_test)
print('Weighted F1', weighted_f1_test)
print('Accuracy', accuracy_test)

labels_sorted = sorted(y_test.unique())
cm_counts = confusion_matrix(y_test, test_pred, labels=labels_sorted)
cm_normalized = confusion_matrix(y_test, test_pred, labels=labels_sorted, normalize='true')

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
im0 = axes[0].imshow(cm_counts, cmap='Blues')
axes[0].set_title('Confusion matrix (counts)')
axes[0].set_xticks(range(len(labels_sorted)))
axes[0].set_yticks(range(len(labels_sorted)))
axes[0].set_xticklabels(labels_sorted)
axes[0].set_yticklabels(labels_sorted)
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')
for i in range(len(labels_sorted)):
    for j in range(len(labels_sorted)):
        axes[0].text(j, i, cm_counts[i, j], ha='center', va='center')

im1 = axes[1].imshow(cm_normalized, cmap='Oranges')
axes[1].set_title('Confusion matrix (row normalized)')
axes[1].set_xticks(range(len(labels_sorted)))
axes[1].set_yticks(range(len(labels_sorted)))
axes[1].set_xticklabels(labels_sorted)
axes[1].set_yticklabels(labels_sorted)
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('True')
for i in range(len(labels_sorted)):
    for j in range(len(labels_sorted)):
        axes[1].text(j, i, round(cm_normalized[i, j], 2), ha='center', va='center')

plt.tight_layout()
plt.show()

precision, recall, f1, support = precision_recall_fscore_support(y_test, test_pred, labels=labels_sorted)
class_report_df = pd.DataFrame({
    'class': labels_sorted,
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'support': support
})

x = np.arange(len(labels_sorted))
width = 0.25
plt.figure(figsize=(7, 4))
plt.bar(x - width, precision, width, label='Precision')
plt.bar(x, recall, width, label='Recall')
plt.bar(x + width, f1, width, label='F1')
plt.xticks(x, labels_sorted)
plt.title('Per class precision, recall and F1 (' + best_name + ')')
plt.legend()
plt.tight_layout()
plt.show()
class_report_df


# ## 11. Error analysis
# 
# Misclassified tweets are extracted for manual inspection. At least five to ten cases should be reviewed and annotated for the challenge categories listed in the lab manual, such as negation, sarcasm, emoji heavy sentiment, hashtag sentiment and mixed sentiment.

error_mask = (test_pred != y_test.values)
error_df = test_df.loc[y_test.index[error_mask], [c for c in [ID_COL, TEXT_COL, TARGET_COL, ENTITY_COL] if c in test_df.columns]].copy()
error_df['prediction'] = test_pred[error_mask]
error_sample = error_df.sample(n=min(10, len(error_df)), random_state=SEED)
error_sample['challenge_type'] = ''
error_sample['likely_reason'] = ''
error_sample['mitigation'] = ''
error_sample.to_csv(os.path.join(OUT, 'error_analysis.csv'), index=False)
error_sample


# ## 12. Product and entity level analysis
# 
# Sentiment distribution and error rate are reported by airline. Entities with fewer than 30 held out tweets are excluded from ranking, following the minimum support rule in the manual.

MIN_SUPPORT = 30
pred_df = test_df[[c for c in [ID_COL, TEXT_COL, TARGET_COL, ENTITY_COL] if c in test_df.columns]].copy()
pred_df['prediction'] = test_pred
pred_df['correct'] = pred_df['prediction'] == pred_df[TARGET_COL]

entity_support = pred_df[ENTITY_COL].value_counts()
valid_entities = entity_support[entity_support >= MIN_SUPPORT].index
filtered_pred = pred_df[pred_df[ENTITY_COL].isin(valid_entities)]

entity_summary = pd.crosstab(filtered_pred[ENTITY_COL], filtered_pred['prediction'], normalize='index').round(3)
entity_summary['N'] = entity_support.loc[entity_summary.index]
entity_summary.to_csv(os.path.join(OUT, 'entity_sentiment_distribution.csv'))
entity_summary

entity_error_rate = filtered_pred.groupby(ENTITY_COL)['correct'].apply(lambda s: 1 - s.mean())

plt.figure(figsize=(7, 4))
entity_error_rate.sort_values(ascending=False).plot(kind='bar', color='#e67e22')
plt.title('Error rate by airline (entities with N >= ' + str(MIN_SUPPORT) + ')')
plt.ylabel('Error rate')
plt.tight_layout()
plt.show()


# ## 13. Save artifacts and reload check
# 
# The selected pipeline and predictions are saved. A small reload check confirms the saved pipeline produces identical predictions to the in memory model, which is required before the pipeline can be trusted for later use.

pred_df.to_csv(os.path.join(OUT, 'test_predictions.csv'), index=False)
joblib.dump(best_model, os.path.join(OUT, 'selected_pipeline.joblib'))

reloaded = joblib.load(os.path.join(OUT, 'selected_pipeline.joblib'))
check_sample = X_test.iloc[:20]
assert np.array_equal(reloaded.predict(check_sample), best_model.predict(check_sample))
print('Reload check passed')

readme_lines = [
    'Lab 05 Tweet Sentiment Analysis',
    'Registration No: 23MID0296',
    'Dataset: Twitter US Airline Sentiment (D2)',
    'Selected model: ' + best_name,
    'Test macro F1: ' + str(round(macro_f1_test, 4)),
    'Test weighted F1: ' + str(round(weighted_f1_test, 4)),
    'Test accuracy: ' + str(round(accuracy_test, 4)),
    'Random seed: ' + str(SEED),
]
with open(os.path.join(OUT, 'README.md'), 'w') as f:
    f.write('\n'.join(readme_lines))
print('README saved')


# ## 14. Acceptance tests
# 
# Structural checks confirming the required columns, split integrity and output artifacts are all present before submission.

assert TEXT_COL in df.columns and TARGET_COL in df.columns
assert df[TARGET_COL].nunique() >= 2
assert not df[TEXT_COL].isna().any()
assert len(set(train_df.index).intersection(set(test_df.index))) == 0
assert all(name in models for name in ['LogisticRegression', 'LinearSVC'])
assert os.path.exists(os.path.join(OUT, 'cv_results.csv'))
assert os.path.exists(os.path.join(OUT, 'selected_pipeline.joblib'))
assert os.path.exists(os.path.join(OUT, 'test_predictions.csv'))
print('Core acceptance tests passed')


# ## 15. Responsible use note
# 
# This model supports aggregate monitoring of airline sentiment on Twitter. It should not be used to identify individual users, infer personal traits, or make employment, credit or eligibility decisions. Usernames and coordinates were excluded from modeling. Results reflect a 2015 Twitter sample and may not represent current social media language or airline performance.