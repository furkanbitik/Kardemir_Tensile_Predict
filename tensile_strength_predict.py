# -*- coding: utf-8 -*-

"""
# Çelik Çekme Dayanımı Tahmini ve EDA

## Bu Not Defterinin amacı, çeliğin bileşimiyle mukavemetini tahmin etmeye çalışmaktır.

# Dataset

Bu veri seti, farklı bileşimlere sahip 312 çeliği temsil etmektedir. Her sütun şunları temsil etmektedir:
* **formula** - Çeliğin formülü
* **c** - Karbon içeriği (%)
* **mn** - Manganez içeriği (%)
* **si** - Silikon içeriği (%)
* **cr** - Krom içeriği (%)
* **ni** - Nikel içeriği (%)
* **mo** - Molibden içeriği (%)
* **v** - Vanadyum içeriği (%)
* **n** - Azot içeriği (%)
* **nb** - Niyobyum içeriği (%)
* **co** - Kobalt içeriği (%)
* **w** - Tungsten içeriği (%)
* **al** - Alüminyum içeriği (%)
* **ti** - Titanyum içeriği (%)
* **yield strength** - Akma mukavemeti
* **tensile strength** - Çekme mukavemeti
* **elongation** - Uzama

# Drive Entegre Etme

## Kütüphaneleri Yükleme
"""
import matplotlib.pyplot as plt
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor, VotingRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import  mean_absolute_error, r2_score
from lazypredict.Supervised import LazyRegressor
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import plotly.express as px
import warnings
from datetime import datetime as dt
import numpy as np
from sklearn.compose import ColumnTransformer
import os
warnings.simplefilter(action="ignore", category= UserWarning)
warnings.simplefilter(action="ignore", category= pd.errors.SettingWithCopyWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)
pd.set_option("display.max_columns", None)

"""## Veri Yükleme"""

df = pd.read_csv('steel_strength.csv')

"""# İlk Analiz"""

df.head()

"""Bu projenin hedefi, çeliğin çekme mukavemeti, yani çeliğin kalıcı olarak şekil değiştirmeden önce uygulanabilecek maksimum çekilmesi.

Kaynak: https://www.cliftonsteel.com/knowledge-center/tensile-and-yield-strength

Uzama ve akma mukavemeti, çekme mukavemeti ile aynı testte ölçüldüğünden, sızıntıyı önlemek için kaldırılmıştır. Ayrıca amaç, çeliğin test edilmeden önce, sadece bileşimi ile çekme mukavemetini tahmin etmektir. Formül, her gözlem için benzersiz değerlere sahip olduğundan da kaldırılmıştır.
"""

df.drop(columns=["formula", "elongation", "yield strength"], inplace=True)

df.shape

df.duplicated().sum()

df.dtypes

df.isna().sum()

df.describe().T

df.nunique()

"""# Veri Analizi

## Histogramlar
"""

df.hist(figsize=(17,12));

"""Buradan, birçok sütunda aykırı değerler olduğunu ve hiçbir sütunun normal dağılım göstermediğini söyleyebiliriz.

## Korelasyon
"""

df.corr().style.background_gradient(axis=None)

df.corrwith(df["tensile strength"])

"""# Veri Temizleme"""

def drop_outliers_IQR(df):

    q1=df.quantile(0.25)

    q3=df.quantile(0.75)

    IQR=q3-q1

    df_without_outliers = df[~((df<(q1-1.5*IQR)) | (df>(q3+1.5*IQR)))].dropna()

    return df_without_outliers

df_without_outliers = drop_outliers_IQR(df)
df_test_row = df_without_outliers.tail(1)
df_without_outliers.drop(df_test_row.index,inplace=True)

print("Initial shape:", df.shape)
print("Shape after dropping outliers:", df_without_outliers.shape)

"""# Veri Bölme"""

X = df_without_outliers.drop(columns=['tensile strength'])
y = df_without_outliers['tensile strength']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

"""# Temel Çizginin Tanımlanması(Baseline)"""

y_mean = df["tensile strength"].mean()
y_pred_mae = [y_mean] * len(df)
acc_baseline = mean_absolute_error(df["tensile strength"], y_pred_mae)
print("Baseline MAE:", round(acc_baseline))

"""# Lazy Regressor

En iyi üç modeli elde etmek için Lazy Regressor kullanılması ve ardından hiperparametreleri ayarlayarak bunları iyileştirmeye çalışılması tercih edildi.
"""

from contextlib import redirect_stdout, redirect_stderr
reg = LazyRegressor(verbose=0,ignore_warnings=True, custom_metric=None )
with open(os.devnull, 'w') as f, redirect_stdout(f), redirect_stderr(f):
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)
models.head()

"""Yani en iyi üç model BagginRegressor, ExtraTreesRegressor ve RandomForestRegressor'dur.

Farklı aralık ölçeğine sahip çok sayıda değer, çok sayıda uç değer ve verilerin normal dağılım göstermemesi nedeniyle StandardScaler'ı da denedim.
"""

X_train_sts, X_test_sts = StandardScaler().fit_transform(X_train), StandardScaler().fit_transform(X_test)
reg = LazyRegressor(verbose=0,ignore_warnings=True, custom_metric=None )
with open(os.devnull, 'w') as f, redirect_stdout(f), redirect_stderr(f):
  models,predictions = reg.fit(X_train_sts, X_test_sts, y_train, y_test)
models.head()

"""Ancak sonuçlar öncekinden çok daha kötüydü.

# Model 1: Random Forest

### Parametre Ayarları:
"""

params = {
    "n_estimators": range(450,1000,100),
    "max_depth": range(20,61,5),
    "criterion": ["squared_error", "absolute_error"],
    "min_samples_split": [2,4],
    "min_samples_leaf": [1,2,4]
}

"""### Model Kurma:"""

model_rf = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    params,
    cv=5,
    n_jobs=-1,
    n_iter=35,
    scoring=["neg_mean_absolute_error", "r2"],
    refit="neg_mean_absolute_error",
    verbose=1
)

"""### Eğitim:"""

model_rf.fit(X_train, y_train)

"""### Sonuçlar:"""

# Eğitim verilerinden özellik adlarını alın
features = X_train.columns
# Modelden önem derecelerini çıkarın
importances = model_rf.best_estimator_.feature_importances_
# Özellik adları ve önem dereceleriyle bir dizi oluşturun
feat_imp = pd.Series(importances, index=features)
# En önemli 10 özelliği çiz
feat_imp.sort_values().plot(kind="barh")
plt.xlabel("Önem")
plt.ylabel("Özellik")
plt.title("Özellik Önem Oranı Tablosu(Feature Importance)")

"""Görünüşe göre, kaldırılabilecek birçok özellik var.

### Modelin değerlendirilmesi:
"""

mae_rf_train = mean_absolute_error(y_train, model_rf.predict(X_train))
mae_rf_test = mean_absolute_error(y_test, model_rf.predict(X_test))

print("Random Forest:")
print("Training Mean Absolute Error:", round(mae_rf_train, 4))
print("Test Mean Absolute Error:", round(mae_rf_test, 4))
print("Baseline Mean Absolute Error:", round(acc_baseline, 4))

r2_rf_train = r2_score(y_train, model_rf.predict(X_train))
r2_rf_test = r2_score(y_test, model_rf.predict(X_test))

print("Training R2:", round(r2_rf_train, 4))
print("Test R2:", round(r2_rf_test, 4))

"""# Model 2: Bagging Regressor"""

params_br = {
    "n_estimators": range(5,50,5),
}

model_br = GridSearchCV(
    BaggingRegressor(random_state=42),
    params_br,
    cv=5,
    n_jobs=-1,
    scoring=["neg_mean_absolute_error", "r2"],
    refit="neg_mean_absolute_error",
    verbose=1
)

model_br.fit(X_train, y_train)

mae_br_train = mean_absolute_error(y_train, model_br.predict(X_train))
mae_br_test = mean_absolute_error(y_test, model_br.predict(X_test))

print("Bagging Regressor:")
print("Training Mean Absolute Error:", round(mae_br_train, 4))
print("Test Mean Absolute Error:", round(mae_br_test, 4))
print("Baseline Mean Absolute Error:", round(acc_baseline, 4))


r2_br_train = r2_score(y_train, model_br.predict(X_train))
r2_br_test = r2_score(y_test, model_br.predict(X_test))

print("Training R2:", round(r2_br_train, 4))
print("Test R2:", round(r2_br_test, 4))

"""# Model 3: ExtraTreesRegressor"""

params_et = {
    "n_estimators": range(100,1001,100),
    "max_depth": range(20,61,5),
    "criterion": ["squared_error", "absolute_error"],
    "min_samples_split": [2,3],
    "min_samples_leaf": [1,2,3]
}

model_et = RandomizedSearchCV(
    ExtraTreesRegressor(random_state=42),
    params_et,
    cv=5,
    n_jobs=-1,
    scoring=["neg_mean_absolute_error", "r2"],
    n_iter=35,
    refit="neg_mean_absolute_error",
    verbose=1
)

model_et.fit(X_train, y_train)

mae_et_train = mean_absolute_error(y_train, model_et.predict(X_train))
mae_et_test = mean_absolute_error(y_test, model_et.predict(X_test))

print("Extra Trees Regressor:")
print("Training Mean Absolute Error:", round(mae_et_train, 4))
print("Test Mean Absolute Error:", round(mae_et_test, 4))
print("Baseline Mean Absolute Error:", round(acc_baseline, 4))

r2_et_train = r2_score(y_train, model_et.predict(X_train))
r2_et_test = r2_score(y_test, model_et.predict(X_test))

print("Training R2:", round(r2_et_train, 4))
print("Test R2:", round(r2_et_test, 4))

"""## Model 4: Voting Regressor"""

pipe = VotingRegressor(estimators=[("rf", model_rf.best_estimator_),
                                  ("et" ,model_et.best_estimator_),
                                  ("br" ,model_br.best_estimator_)])

params_vr = {
    "weights": [None, [2,3,2], [2,1,2]]
}

model_vr = GridSearchCV(
    pipe,
    params_vr,
    cv=5,
    n_jobs=-1,
    scoring=["neg_mean_absolute_error", "r2"],
    refit="neg_mean_absolute_error",
    verbose=1
)

model_vr.fit(X_train, y_train)

mae_vr_train = mean_absolute_error(y_train, model_vr.predict(X_train))
mae_vr_test = mean_absolute_error(y_test, model_vr.predict(X_test))

print("Voting Regressor:")
print("Training Mean Absolute Error:", round(mae_vr_train, 4))
print("Test Mean Absolute Error:", round(mae_vr_test, 4))
print("Baseline Mean Absolute Error:", round(acc_baseline, 4))

r2_vr_train = r2_score(y_train, model_vr.predict(X_train))
r2_vr_test = r2_score(y_test, model_vr.predict(X_test))

print("Training R2:", round(r2_vr_train, 4))
print("Test R2:", round(r2_vr_test, 4))

"""# Sonuç"""

fig = px.bar(y=["Extra Trees", "Random Forest", "Bagging", "Voting", "Baseline"],
             x=[mae_et_test, mae_rf_test, mae_br_test, mae_vr_test, acc_baseline],
             color=["Extra Trees", "Random Forest", "Bagging", "Voting", "Baseline"],
             color_discrete_map={"Extra Trees": "orange",
                                "Random Forest": "green",
                                "Bagging": "blue",
                                "Voting": "grey",
                                "Baseline": "pink"},
            title="Mean Absolute Error comparison (lower is better)")
fig.update_layout(yaxis={'categoryorder':'total descending'}, xaxis_title="MAE", yaxis_title="Models")
fig.show()

fig = px.bar(y=["Extra Trees", "Random Forest", "Bagging", "Voting"],
             x=[r2_et_test, r2_rf_test, r2_br_test, r2_vr_test],
             color=["Extra Trees", "Random Forest", "Bagging", "Voting"],
             color_discrete_map={"Extra Trees": "orange",
                                "Random Forest": "green",
                                "Bagging": "blue",
                                "Voting": "grey",},
            title="R2-Score comparison (higher is better)")
fig.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_title="R2-score", yaxis_title="Models")
fig.show()

"""* Ortalama mutlak hata puanı ve R2 puanı için en iyi model Extra Trees Regressor.


# Yeni Tahmin
## Tensile Strength değerini modele vermediğimiz bir yeni veri satırı ile modelin son durumunu görme
"""


df_test_row.head()
gercek = df_test_row['tensile strength'].values[0]
df_test_row.drop(columns=['tensile strength'], inplace=True)
predict = model_et.predict(df_test_row)
# Sonuçları yazdıralım

print("--- Tek Satırlık Test Verisiyle Tahmin ---")
print(f"\nSeçilen Test Satırının Özellikleri:\n{df_test_row.to_string(index=False)}")
print("-" * 40)
print(f"Bu satırın gerçek 'tensile strength' değeri: {gercek:.2f}")
print(f"Modelin Tahmin Ettiği 'tensile strength' değeri: {predict[0]:.2f}")

