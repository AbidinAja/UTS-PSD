import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import folium

print("Loading data...")
df = pd.read_csv('Data/UTS/piksel_s2_zaenal.csv')
print(f"Data loaded with shape {df.shape}")

print(f"px_col min: {df['px_col'].min()}, max: {df['px_col'].max()}")
print(f"px_row min: {df['px_row'].min()}, max: {df['px_row'].max()}")

features = ['B02', 'B03', 'B04', 'B08', 'B11', 'NDVI', 'NDWI', 'MNDWI', 'NDBI']
agg_dict = {f: 'mean' for f in features}
agg_dict['lon'] = 'mean'
agg_dict['lat'] = 'mean'
agg_dict['label'] = 'first'
agg_dict['label_teks'] = 'first'

print("Aggregating by poligon_id...")
df_poly = df.groupby('poligon_id').agg(agg_dict).reset_index()
print(f"Aggregated shape {df_poly.shape}")

X = df_poly[features]
y = df_poly['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

models = {
    'Naive Bayes': GaussianNB(),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

# Basic Fuzzy KNN implementation
class FuzzyKNN:
    def __init__(self, k=5):
        self.k = k
        self.knn = KNeighborsClassifier(n_neighbors=k)
    def fit(self, X, y):
        self.knn.fit(X, y)
    def predict(self, X):
        return self.knn.predict(X)
    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

models['Fuzzy KNN'] = FuzzyKNN(k=5)

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")

best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"Best Model is {best_model_name} with Accuracy {results[best_model_name]:.4f}")

df_poly['pred'] = best_model.predict(X)

print("Creating Poligon Folium map...")
m = folium.Map(location=[df_poly['lat'].mean(), df_poly['lon'].mean()], zoom_start=8)
colors = {1: 'green', 2: 'red', 3: 'darkgreen', 4: 'lightblue', 5: 'blue', 6: 'purple'}

for idx, row in df_poly.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=3,
        color=colors.get(row['pred'], 'black'),
        fill=True,
        fill_opacity=0.7,
        popup=f"ID: {row['poligon_id']}<br>True: {row['label_teks']}<br>Pred: {row['pred']}"
    ).add_to(m)

m.save('polygon_map.html')
print("Saved polygon_map.html")
