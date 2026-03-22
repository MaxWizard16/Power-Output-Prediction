
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from statsmodels.tsa.stattools import adfuller



df = pd.read_csv("Real_Time_Data.csv")

print("Rows, Columns:", df.shape)
print("Total Values:", df.size)
print("\nNull Values:\n", df.isnull().sum())
print("\nDataset Description:\n")
print(df.describe())



df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.sort_values('Datetime').reset_index(drop=True)

print("\nTime Range:")
print("Start:", df['Datetime'].min())
print("End  :", df['Datetime'].max())

print("\nSampling Interval Check:")
print(df['Datetime'].diff().value_counts().head(10))



plt.figure(figsize=(12, 4))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Value Heatmap")
plt.show()



plt.figure(figsize=(8, 4))
sns.kdeplot(df['Power'], fill=True)
plt.title("Power Distribution (KDE)")
plt.xlabel("Power")
plt.show()


plt.figure(figsize=(6, 4))
sns.boxplot(x=df['Power'])
plt.title("Power Outlier Detection")
plt.show()



plt.figure(figsize=(14, 4))
plt.plot(df['Datetime'], df['Power'])
plt.title("Power Over Time")
plt.xlabel("Time")
plt.ylabel("Power")
plt.show()


df['Power_roll_60'] = df['Power'].rolling(60).mean()

plt.figure(figsize=(14, 4))
plt.plot(df['Datetime'], df['Power'], alpha=0.4, label='Raw')
plt.plot(df['Datetime'], df['Power_roll_60'], color='red', label='60-min Avg')
plt.legend()
plt.title("Power Trend with Rolling Mean")
plt.show()



lags = [1, 5, 15, 30, 60]

for lag in lags:
    df[f'Power_lag_{lag}'] = df['Power'].shift(lag)

lag_corr = df[['Power'] + [f'Power_lag_{l}' for l in lags]].corr()

plt.figure(figsize=(6, 5))
sns.heatmap(lag_corr, annot=True, cmap='coolwarm')
plt.title("Lag vs Power Correlation")
plt.show()



corr = df.corr(numeric_only=True)

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr,
    cmap='coolwarm',
    center=0,
    linewidths=0.5,
    annot=False
)
plt.title("Feature Correlation Heatmap")
plt.show()

print("\nCorrelation with Power:")
print(corr['Power'].sort_values(ascending=False))



adf_result = adfuller(df['Power'].dropna())
print("\nADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])



df['Power_target'] = df['Power'].shift(-15)

df = df.dropna().reset_index(drop=True)

X = df.drop(columns=['Power', 'Power_target', 'Datetime'])
y = df['Power_target']



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)



rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=30,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)



y_pred = rf.predict(X_test)

print("\n15-Minute Ahead Power Prediction (Random Forest)")
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("R²  :", r2_score(y_test, y_pred))



plt.figure(figsize=(10, 4))
plt.plot(y_test.values[:300], label='Actual Power')
plt.plot(y_pred[:300], '--', label='Predicted Power')
plt.title("15-Minute Ahead Power Prediction")
plt.xlabel("Time Steps")
plt.ylabel("Power")
plt.legend()
plt.show()



importances = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(8, 4))
importances.plot(kind='bar')
plt.title("Random Forest Feature Importance")
plt.show()
