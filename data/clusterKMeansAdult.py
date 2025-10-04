# Clustering incremental con River (simulación de flujo) + evaluación/visualización batch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from river import cluster, preprocessing, stream
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans as SKKMeans  # solo si quisieras comparar (opcional)
from river import compose
# -------------------------------
# Configuración
csv_path = "data/adult.csv"   # Ajusta si tu CSV está en otra ruta
sample_size = None           # set None para usar todo (puede tardar mucho)
n_clusters = 2
random_state = 1
# -------------------------------

# --- 1) Cargar (y muestrear para eficiencia si se desea) ---
df_full = pd.read_csv(csv_path)

if sample_size is not None:
    df = df_full.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
else:
    df = df_full.copy()

# Guardamos income para análisis posterior (no se usa para entrenamiento del cluster)
y_income = df['income'].astype(str).reset_index(drop=True)

# Características (sin la etiqueta)
X_df = df.drop(columns=['income'])

# --- 2) Crear flujo (generador) que emula llegada fila a fila ---
# Convertimos cada fila a dict (River espera dicts)
data_stream = (row.dropna().to_dict() for _, row in X_df.iterrows())

# --- 3) Definir pipeline incremental de River:
# Primero OneHotEncoder (categorías -> numérico), luego StandardScaler y KMeans incremental
model = (
    #compose.Discard('capital.loss','capital.gain', 'hours.per.week','age') 
    preprocessing.OneHotEncoder() |
    preprocessing.StandardScaler() |
    cluster.KMeans(n_clusters=2, sigma=0.5, halflife=2, seed=42)
    #cluster.DBSTREAM(clustering_threshold=0.005, minimum_weight=4, fading_factor= 0.05)
)

# --- 4) Bucle streaming: predecir asignación, guardar y aprender (in-place) ---
assignments = []  # guarda ID del clúster por instancia

for i, (x_dict) in enumerate(data_stream, start=1):
    # predict_one puede devolver None si el modelo aún no tiene centroides inicializados
    cluster_id = model.predict_one(x_dict)
    if cluster_id is None:
        # asignamos -1 mientras no haya clusters definidos
        assignments.append(-1)
    else:
        assignments.append(cluster_id)
    # aprender in-place (IMPORTANTE: no reasignar model = model.learn_one(...))
    model.learn_one(x_dict)
    # opcional: progreso
    if i % 1000 == 0:
        print(f"Procesadas {i} instancias...")

print("Entrenamiento incremental finalizado. Instancias procesadas:", len(assignments))
#print(model['KMeans'].centers)

# --- 5) Post-procesado para evaluación/visualización (batch sobre la muestra usada) ---
# Reproducimos el mismo preprocesado que hizo River para obtener la matriz numérica
# (Usamos pandas.get_dummies + StandardScaler para evaluación/silhouette/PCA)
X_batch_encoded = pd.get_dummies(X_df, drop_first=True)
scaler = StandardScaler()
X_batch_scaled = scaler.fit_transform(X_batch_encoded.values)

# Para las instancias iniciales asignadas a -1 (mientras no había clusters), 
# podemos volver a asignarlas usando el KMeans final de sklearn entrenado con los mismos centroids
# pero en muchos casos no es necesario; para silhouette debemos tener etiquetas válidas.
# Si hay -1, reasignamos usando los centroides del modelo River (si accesibles) o usando sklearn KMeans fit.
if any([a == -1 for a in assignments]):
    # Reajustamos un KMeans sklearn con las mismas n_clusters sobre la representación batch
    km_refit = SKKMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    km_refit.fit(X_batch_scaled)
    assignments = km_refit.labels_.tolist()
    print("Se reasignaron clusters usando KMeans (batch) para obtener etiquetas completas.")
else:
    # assignments ya tiene ids válidos
    assignments = np.array(assignments)

# --- 6) Métrica: Silhouette (requiere >1 clusters y etiquetas válidas) ---
# if n_clusters > 1:
#     silhouette_avg = silhouette_score(X_batch_scaled, assignments)
# else:
#     silhouette_avg = float('nan')

# print(f"\nSilhouette Score (muestra): {silhouette_avg:.4f}")

# --- 7) Preparar DataFrame para gráficas (PCA) ---
pca = PCA(n_components=2, random_state=random_state)
X_pca = pca.fit_transform(X_batch_scaled)
plot_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "Cluster": assignments.astype(str),
    "Income": y_income.values
})

# --- 8) Gráfico: distribución de income por clúster ---
plt.figure(figsize=(10, 6))
sns.countplot(data=plot_df, x='Cluster', hue='Income', palette='viridis')
plt.title('Distribución de Ingresos ("income") por Clúster (KMeans incremental)')
plt.xlabel('ID del Clúster')
plt.ylabel('Conteo de Instancias')
plt.legend(title='Income')
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

# --- 9) Gráfico PCA coloreado por clúster ---
plt.figure(figsize=(12, 8))
sns.scatterplot(data=plot_df, x='PC1', y='PC2', hue='Cluster', palette='Set1', s=20, alpha=0.7)
plt.title('Proyección PCA de clústeres (KMeans incremental en muestra)')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
plt.legend(title='Cluster')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- 10) Mostrar tabla de distribución por cluster e income ---
results_df = pd.DataFrame({
    'Cluster': assignments,
    'Income': y_income.values
})
print("\nDistribución de instancias por clúster (con proporciones por income):")
for cid in sorted(np.unique(assignments)):
    sub = results_df[results_df['Cluster'] == cid]
    counts = sub['Income'].value_counts(normalize=True).round(3)
    print(f"\nCluster {cid} (N={len(sub)}):")
    print(counts.to_string())
