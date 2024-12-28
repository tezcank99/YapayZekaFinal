# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 00:29:49 2024

@author: Tezcank5
"""

import pandas as pd

data_path = '/Users/Tezcank5/Desktop/ÜNİ/yapayzeka/ist.csv'
df = pd.read_csv(data_path)

# Veriyi gözden geçirme
print(df.head())
print(df.info())

dosya_yolu = r'C:\Users\Tezcank5\Desktop\ÜNİ/yapayzeka/ist.csv'

# Değişiklikleri kaydetme
df.to_csv(dosya_yolu, index=False)
from geopy.distance import geodesic


# Taksim Meydanı'nın koordinatları 
taksim_coordinates = (41.0369, 28.9850)

# Mesafeyi hesaplama fonksiyonu
def calculate_distance(row, reference_point):
    property_coordinates = (row['enlem'], row['boylam'])
    return geodesic(property_coordinates, reference_point).kilometers

# Mesafe özelliği ekleme
df['taksim_mesafe'] = df.apply(calculate_distance, reference_point=taksim_coordinates, axis=1)

# Güncellenen veriyi kaydetme
df.to_csv(dosya_yolu, index=False)

# Yeni özellikleri ve mesafeyi içeren veriyi görüntüleme
print(df[['enlem', 'boylam', 'taksim_mesafe']].head())


print(df.dtypes)

# Eksik değerleri yeniden kontrol etme
# Sayısal kolonları belirleme
numeric_columns = ['fiyat', 'enlem', 'boylam', 'yorum_sayisi','taksim_mesafe']

# Aykırı değerleri belirlemek için IQR (Interquartile Range) yöntemi
Q1 = df[numeric_columns].quantile(0.25)
Q3 = df[numeric_columns].quantile(0.75)
IQR = Q3 - Q1

# Aykırı değerleri filtreleme
df = df[~((df[numeric_columns] < (Q1 - 1.5 * IQR)) | (df[numeric_columns] > (Q3 + 1.5 * IQR))).any(axis=1)]
# Sütun adlarındaki boşlukları temizleme
df.columns = df.columns.str.strip()

# Kategorik değişkenleri kodlama
# Kategorik değişkenleri belirleme (doğru sütun adlarını kullanarak)
categorical_columns = ['mahalle', 'oda_tipi']

# Kategorik değişkenleri one-hot encoding ile kodlama
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Veri setinin ilk birkaç satırını görüntüleme
print(df_encoded.head())
# Yorum sayısı kolonunda sıfır olan verileri bulma

# Metin sütunlarını çıkarma
df_encoded = df_encoded.drop(columns=['isim'])

from sklearn.model_selection import train_test_split

# Bağımlı ve bağımsız değişkenleri ayırma
X = df_encoded.drop(columns=['fiyat'])  # Bağımsız değişkenler
y = df_encoded['fiyat']  # Bağımlı değişken

# Veriyi eğitim ve test setlerine bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Metin sütunlarını çıkarma

# Test seti üzerinde tahmin yapma
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Modeli oluşturma ve eğitme
model = LinearRegression()
model.fit(X_train, y_train)

# Test seti üzerinde tahmin yapma
y_pred = model.predict(X_test)

# Performans metrikleri
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


from sklearn.metrics import mean_absolute_error
import numpy as np

# Mean Absolute Error (MAE) hesaplama
mae = mean_absolute_error(y_test, y_pred)

# Root Mean Squared Error (RMSE) hesaplama
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")
print("-------------------------------------------")
from sklearn.model_selection import GridSearchCV

# Hiperparametre grid'i
param_grid = {
    'fit_intercept': [True, False],
    'copy_X': [True, False],
    'n_jobs': [None, -1],
    'positive': [True, False]
}

# GridSearchCV tanımlama
grid_search = GridSearchCV(LinearRegression(), param_grid, cv=5)

# Modeli eğitim verileriyle eğitme
grid_search.fit(X_train, y_train)

# En iyi parametreler
print(f"En iyi parametreler: {grid_search.best_params_}")

# En iyi modeli kullanarak tahmin yapma
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Performans metrikleri
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
print("-------------------------------------------")


from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# Hiperparametre grid'i
param_grid = {
    'alpha': [0.1, 1.0, 10.0, 100.0]
}

# GridSearchCV tanımlama
ridge_grid_search = GridSearchCV(Ridge(), param_grid, cv=5)

# Modeli eğitim verileriyle eğitme
ridge_grid_search.fit(X_train, y_train)

# En iyi parametreler
print(f"En iyi parametreler (Ridge): {ridge_grid_search.best_params_}")

# En iyi modeli kullanarak tahmin yapma
ridge_best_model = ridge_grid_search.best_estimator_
y_pred_ridge = ridge_best_model.predict(X_test)

# Performans metrikleri
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print(f"Mean Squared Error (Ridge): {mse_ridge}")
print(f"R-squared (Ridge): {r2_ridge}")

print("-------------------------------------------")

from sklearn.linear_model import Lasso

# Hiperparametre grid'i
param_grid = {
    'alpha': [0.01, 0.1, 1.0, 10.0]
}

# GridSearchCV tanımlama
lasso_grid_search = GridSearchCV(Lasso(), param_grid, cv=5)

# Modeli eğitim verileriyle eğitme
lasso_grid_search.fit(X_train, y_train)

# En iyi parametreler
print(f"En iyi parametreler (Lasso): {lasso_grid_search.best_params_}")

# En iyi modeli kullanarak tahmin yapma
lasso_best_model = lasso_grid_search.best_estimator_
y_pred_lasso = lasso_best_model.predict(X_test)

# Performans metrikleri
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

print(f"Mean Squared Error (Lasso): {mse_lasso}")
print(f"R-squared (Lasso): {r2_lasso}")
print("-------------------------------10 dk  bekle-------------------------------")
from xgboost import XGBRegressor

# Hiperparametre grid'i
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0]
}

# GridSearchCV tanımlama
xgb_grid_search = GridSearchCV(XGBRegressor(), param_grid, cv=5)

# Modeli eğitim verileriyle eğitme
xgb_grid_search.fit(X_train, y_train)

# En iyi parametreler
print(f"En iyi parametreler (XGBoost): {xgb_grid_search.best_params_}")

# En iyi modeli kullanarak tahmin yapma
xgb_best_model = xgb_grid_search.best_estimator_
y_pred_xgb = xgb_best_model.predict(X_test)

# Performans metrikleri
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f"Mean Squared Error (XGBoost): {mse_xgb}")
print(f"R-squared (XGBoost): {r2_xgb}")


# Hata hesaplama (mutlak fark)
errors = np.abs(y_test - y_pred)

# 150 TL'ye kadar olan hata paylarını doğru kabul etme
tolerated_error = (errors <= 150)  # Hata 150 TL'yi geçmediğinde doğru kabul edilecek

# Toleranslı başarı oranı (0-150 TL hata aralığında)
tolerated_accuracy = np.mean(tolerated_error)

# Sonuçları yazdırma
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
print(f"Toleranslı başarı oranı (0-150 TL arası hata): {tolerated_accuracy * 100:.2f}%")


# 150 TL'nin altındaki hataları 0 kabul edip, geri kalanları normal MSE hesaplama gibi
tolerated_errors_mse = np.where(tolerated_error, 0, errors ** 2)
tolerated_mse = np.mean(tolerated_errors_mse)
print(f"Toleranslı Mean Squared Error (150 TL hata toleranslı): {tolerated_mse}")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Gerçek ve tahmin edilen değerler
y_true = y_test  # Gerçek değerler
y_pred = y_pred  # Tahmin edilen değerler

# R-squared (R²) hesaplama
r2 = r2_score(y_true, y_pred)
print(f"R-squared: {r2:.4f}")

# Mean Squared Error (MSE) hesaplama
mse = mean_squared_error(y_true, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

# Root Mean Squared Error (RMSE) hesaplama
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse:.4f}")

# Mean Absolute Error (MAE) hesaplama
mae = mean_absolute_error(y_true, y_pred)
print(f"Mean Absolute Error: {mae:.4f}")

# Gerçek vs Tahmin Görselleştirme
plt.figure(figsize=(10, 6))

# Scatter plot: Gerçek vs Tahmin
plt.subplot(1, 2, 1)
plt.scatter(y_true, y_pred, alpha=0.6)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color='red', linestyle='--', lw=2)
plt.xlabel('Gerçek Fiyatlar')
plt.ylabel('Tahmin Edilen Fiyatlar')
plt.title('Gerçek vs Tahmin Fiyatları')

# Hata Görselleştirme: Gerçek ve Tahmin Farkları
errors = y_true - y_pred
plt.subplot(1, 2, 2)
plt.scatter(y_true, errors, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--', lw=2)
plt.xlabel('Gerçek Fiyatlar')
plt.ylabel('Hata (Gerçek - Tahmin)')
plt.title('Tahmin Hataları')

plt.tight_layout()
plt.show()


import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Hata toleransı
tolerance = 150

# Toleranslı metrik hesaplama fonksiyonu
def calculate_tolerated_metrics(y_true, y_pred, tolerance):
    # Hataları hesapla
    errors = np.abs(y_true - y_pred)

    # Toleranslı hata oranı
    tolerated_error = (errors <= tolerance)
    tolerated_accuracy = np.mean(tolerated_error)

    # Toleranslı MSE
    tolerated_errors_mse = np.where(tolerated_error, 0, errors ** 2)
    tolerated_mse = np.mean(tolerated_errors_mse)

    # Toleranslı MAE (0 olanları hesaba katmamak için yeniden hesaplama)
    tolerated_mae = np.mean(errors[~tolerated_error])

    return tolerated_accuracy * 100, tolerated_mse, tolerated_mae

# Toleranslı metrikleri hesaplama
model_names = ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'XGBoost', 'Random Forest']
y_preds = [y_pred, y_pred_ridge, y_pred_lasso, y_pred_xgb]  # Model tahminleri
results = []

for model_name, y_pred_model in zip(model_names, y_preds):
    tolerated_accuracy, tolerated_mse, tolerated_mae = calculate_tolerated_metrics(y_test, y_pred_model, tolerance)
    rmse = np.sqrt(tolerated_mse)  # Toleranslı RMSE

    # Performans metriklerini sakla
    results.append({
        'Model': model_name,
        'Tolerated Accuracy (%)': tolerated_accuracy,
        'Tolerated MSE': tolerated_mse,
        'Tolerated RMSE': rmse,
        'Tolerated MAE': tolerated_mae
    })

# Sonuçları tabloya dökme
results_df = pd.DataFrame(results)

# Tabloyu yazdırma
print(results_df)

# CSV olarak kaydetme
results_df.to_csv('tolerated_metrics_results.csv', index=False)
