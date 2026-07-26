import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor

audit_df = pd.read_csv("C:/codes/Course---Predictive-Analysis/Lab-1/housing.csv")
audit_df.shape

audit = pd.DataFrame({
    'dtype': audit_df.dtypes.astype(str),
    'missing_n': audit_df.isna().sum(),
    'missing_pct': 100 * audit_df.isna().mean(),
    'unique_n': audit_df.nunique(dropna=False)
})
audit

audit_df.duplicated().sum()

audit_df['median_house_value'].describe()

plt.figure(figsize=(8,5))
sns.histplot(audit_df['median_house_value'], kde=True)
plt.xlabel("Median House Value")
plt.title("Target Distribution")
plt.show()

plt.figure(figsize=(8,4))
audit['missing_pct'].plot(kind='bar')
plt.ylabel("Missing %")
plt.title("Missing Value Profile")
plt.show()

plt.figure(figsize=(9,7))
sns.heatmap(audit_df.select_dtypes(include=np.number).corr(), cmap='coolwarm', center=0, annot=True, fmt='.2f')
plt.title("Correlation Matrix")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12,4.5))
axes[0].scatter(audit_df['median_income'], audit_df['median_house_value'], alpha=0.3)
axes[0].set_xlabel("Median Income")
axes[0].set_ylabel("Median House Value")
axes[1].scatter(audit_df['housing_median_age'], audit_df['median_house_value'], alpha=0.3)
axes[1].set_xlabel("Housing Median Age")
axes[1].set_ylabel("Median House Value")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(data=audit_df, x='ocean_proximity', y='median_house_value')
plt.title("Ocean Proximity vs Median House Value")
plt.show()

results = []

df_base = audit_df.dropna()
X_base = df_base[['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income']]
y_base = df_base['median_house_value']
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_base, y_base, test_size=0.2, random_state=42)

naive = DummyRegressor(strategy='mean')
naive.fit(X_train_b, y_train_b)
naive_pred = naive.predict(X_test_b)

print("Naive MAE: ", mean_absolute_error(y_test_b, naive_pred))
print("Naive RMSE: ", np.sqrt(mean_squared_error(y_test_b, naive_pred)))

results.append({'Model': 'Naive Baseline', 'MAE': mean_absolute_error(y_test_b, naive_pred), 'MSE': mean_squared_error(y_test_b, naive_pred), 'RMSE': np.sqrt(mean_squared_error(y_test_b, naive_pred)), 'R2': r2_score(y_test_b, naive_pred), 'Train_RMSE': np.nan})

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("C:/codes/Course---Predictive-Analysis/Lab-1/housing.csv")
df.head()

df = df.dropna()

X = df[['median_income']]
y = df[['median_house_value']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

model = LinearRegression()
model.fit (X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("R² Score: ", r2)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("MAE: ", mae)

train_pred = model.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))

results.append({'Model': 'Simple Linear Regression', 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2, 'Train_RMSE': train_rmse})

if X.shape[1] == 1:
    plt.scatter(X_test, y_test)
    plt.plot(X_test, predictions)
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.title("Linear Regression")
    plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

df.head()

df = df.dropna()

X = df[['longitude',
        'latitude',
        'housing_median_age',
        'total_rooms',
        'total_bedrooms',
        'population',
        'households',
        'median_income']]
y = df['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("R² Score: ", r2)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("MAE: ", mae)

train_pred = model.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))

results.append({'Model': 'Multi-Linear Regression', 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2, 'Train_RMSE': train_rmse})

coef = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

print("\nCoefficients:")
print(coef)

print("\nIntercept:", model.intercept_)

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red', linewidth=2)

plt.xlabel("Actual House Value")
plt.ylabel("Predicted House Value")
plt.title("Actual vs Predicted Values")
plt.show()

residuals = y_test - y_pred

plt.figure(figsize=(8,6))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')

plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree

df.head()

X = df[['longitude',
        'latitude',
        'housing_median_age',
        'total_rooms',
        'total_bedrooms',
        'population',
        'households',
        'median_income']]

y = df['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeRegressor(random_state=42, max_depth=5)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("R² Score: ", r2)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("MAE: ", mae)

train_pred = model.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))

results.append({'Model': 'Decision Tree', 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2, 'Train_RMSE': train_rmse})

plt.figure(figsize=(20,10))
plot_tree(
    model,
    feature_names=X.columns,
    filled=True,
    rounded=True,
    fontsize=8
)
plt.show()

from sklearn.preprocessing import PolynomialFeatures

df.head()

X = df[['median_income']]
y = df['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()

model.fit(X_train_poly, y_train)

y_pred = model.predict(X_test_poly)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("R² Score: ", r2)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("MAE: ", mae)

train_pred = model.predict(X_train_poly)
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))

results.append({'Model': 'Polynomial Regression', 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2, 'Train_RMSE': train_rmse})

X_grid = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
X_grid_poly = poly.transform(X_grid)

plt.figure(figsize=(8,6))
plt.scatter(X, y, color='blue', alpha=0.3, label='Data')
plt.plot(X_grid,
         model.predict(X_grid_poly),
         color='red',
         linewidth=3,
         label='Polynomial Regression')

plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.title("Polynomial Regression (Degree = 2)")
plt.legend()
plt.show()

from sklearn.ensemble import RandomForestRegressor

df.head()

X = df[['longitude',
        'latitude',
        'housing_median_age',
        'total_rooms',
        'total_bedrooms',
        'population',
        'households',
        'median_income']]

y = df['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("R² Score : ", r2)
print("MSE : ", mse)
print("RMSE : ", rmse)
print("MAE : ", mae)

train_pred = model.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))

results.append({'Model': 'Random Forest', 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2, 'Train_RMSE': train_rmse})

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
})

feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8,5))
plt.bar(feature_importance['Feature'], feature_importance['Importance'])
plt.xticks(rotation=45)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red',
         linewidth=2)

plt.xlabel("Actual House Value")
plt.ylabel("Predicted House Value")
plt.title("Random Forest Regression: Actual vs Predicted")
plt.show()

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('RMSE').reset_index(drop=True)
results_df

from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline

df_cv = audit_df.dropna()
X_multi = df_cv[['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income']]
y_multi = df_cv['median_house_value']
X_single = df_cv[['median_income']]
y_single = df_cv['median_house_value']

cv = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = {'mae': 'neg_mean_absolute_error', 'rmse': 'neg_root_mean_squared_error', 'r2': 'r2'}

cv_results = []

scores = cross_validate(LinearRegression(), X_single, y_single, cv=cv, scoring=scoring)
cv_results.append({'Model': 'Simple Linear Regression', 'CV_MAE': -scores['test_mae'].mean(), 'CV_RMSE': -scores['test_rmse'].mean(), 'CV_R2': scores['test_r2'].mean()})

scores = cross_validate(LinearRegression(), X_multi, y_multi, cv=cv, scoring=scoring)
cv_results.append({'Model': 'Multi-Linear Regression', 'CV_MAE': -scores['test_mae'].mean(), 'CV_RMSE': -scores['test_rmse'].mean(), 'CV_R2': scores['test_r2'].mean()})

scores = cross_validate(DecisionTreeRegressor(max_depth=5, random_state=42), X_multi, y_multi, cv=cv, scoring=scoring)
cv_results.append({'Model': 'Decision Tree', 'CV_MAE': -scores['test_mae'].mean(), 'CV_RMSE': -scores['test_rmse'].mean(), 'CV_R2': scores['test_r2'].mean()})

poly_pipe = Pipeline([('poly', PolynomialFeatures(degree=2)), ('model', LinearRegression())])
scores = cross_validate(poly_pipe, X_single, y_single, cv=cv, scoring=scoring)
cv_results.append({'Model': 'Polynomial Regression', 'CV_MAE': -scores['test_mae'].mean(), 'CV_RMSE': -scores['test_rmse'].mean(), 'CV_R2': scores['test_r2'].mean()})

scores = cross_validate(RandomForestRegressor(n_estimators=100, random_state=42), X_multi, y_multi, cv=cv, scoring=scoring, n_jobs=-1)
cv_results.append({'Model': 'Random Forest', 'CV_MAE': -scores['test_mae'].mean(), 'CV_RMSE': -scores['test_rmse'].mean(), 'CV_R2': scores['test_r2'].mean()})

cv_results_df = pd.DataFrame(cv_results).sort_values('CV_RMSE').reset_index(drop=True)
cv_results_df

from sklearn.model_selection import GridSearchCV

param_grid = {'max_depth': [3, 4, 5, 6, 8, 10, 12, None]}
grid = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1)
grid.fit(X_multi, y_multi)

print("Best max_depth: ", grid.best_params_)
print("Best CV RMSE: ", -grid.best_score_)

gap_df = results_df[results_df['Model'] != 'Naive Baseline'][['Model', 'Train_RMSE', 'RMSE']].copy()
gap_df['Gap'] = gap_df['RMSE'] - gap_df['Train_RMSE']
gap_df

residuals = y_test - y_pred

plt.figure(figsize=(8,6))
sns.histplot(residuals, kde=True)
plt.xlabel("Residual")
plt.title("Residual Distribution: Random Forest")
plt.show()

segment_df = X_test.copy()
segment_df['Actual'] = y_test
segment_df['Predicted'] = y_pred
segment_df['AbsError'] = np.abs(segment_df['Actual'] - segment_df['Predicted'])
segment_df['ocean_proximity'] = audit_df.loc[segment_df.index, 'ocean_proximity']

price_segment = pd.qcut(segment_df['Actual'], q=3, labels=['Low', 'Mid', 'High'])
segment_df.groupby(price_segment)['AbsError'].mean()

segment_df.groupby('ocean_proximity')['AbsError'].mean().sort_values()

import joblib

joblib.dump(model, '23MID0296_Lab01_Model.joblib')
results_df.to_csv('23MID0296_Lab01_Results.csv', index=False)
cv_results_df.to_csv('23MID0296_Lab01_CV_Results.csv', index=False)