---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# 3. Modeling Klasifikasi (Naive Bayes vs KNN)

Kini kita mulai membangun model kecerdasan buatan. Kita melatih data ke dalam dua model klasik. Karena Jupyter book memisahkan memori per halaman, mari kita memuat data dan *preprocess* terlebih dahulu di sel tersembunyi.

```{code-cell} ipython3
:tags: [hide-cell]

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('../../Data/UTS/piksel_s2_zaenal.csv')
fitur = ['B02', 'B03', 'B04', 'B08', 'B11', 'NDVI', 'NDWI', 'MNDWI', 'NDBI']
df_bersih = df.dropna(subset=fitur + ['label_teks']).copy()

X = df_bersih[fitur]
y = df_bersih['label_teks']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 5.1 Naive Bayes (Gaussian)
Naive Bayes adalah algoritma berbasis probabilitas yang berasumsi bahwa setiap fitur independen satu sama lain.

```{code-cell} ipython3
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)
y_pred_nb = nb_model.predict(X_test_scaled)

print("=== HASIL EVALUASI NAIVE BAYES ===")
print(f"Akurasi Keseluruhan : {accuracy_score(y_test, y_pred_nb)*100:.2f}%\\n")
print(classification_report(y_test, y_pred_nb))
```

### 5.2 K-Nearest Neighbors (KNN)
KNN menentukan kelas lahan dengan cara mencari tetangga titik-titik data terdekat (kita gunakan nilai tetangga K=5).

```{code-cell} ipython3
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)

print("=== HASIL EVALUASI KNN ===")
print(f"Akurasi Keseluruhan : {accuracy_score(y_test, y_pred_knn)*100:.2f}%\\n")
print(classification_report(y_test, y_pred_knn))
```

**Analisis Evaluasi:** Anda dapat menentukan algoritma manakah yang berkinerja lebih tinggi untuk memecahkan kasus pemetaan ini dengan melihat parameter `Akurasi` dan `F1-score` yang dicetak di atas.
