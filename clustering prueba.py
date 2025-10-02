import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from river import compose
from river import preprocessing
from river import cluster
from river import stream

# --- 1. CONFIGURACIÓN Y CARGA DE DATOS ---
data = fetch_california_housing(as_frame=True)
df = data.frame
target_name = 'MedHouseVal' 
n_clusters = 3
seed_value = 42

# **CORRECCIÓN: Excluir la variable objetivo para evitar data leakage**
features = [col for col in df.columns if col != target_name]
X_df = df[features].copy()  # Solo características, sin el target

print("Características usadas para clustering:")
print(features)
print(f"\nDimensión de datos para clustering: {X_df.shape}")

X_river_format = X_df.to_dict(orient='records') 

# --- 2. DEFINICIÓN Y ENTRENAMIENTO DEL PIPELINE (RIVER) ---

model_clustering = (
    preprocessing.StandardScaler() |
    cluster.KMeans(n_clusters=n_clusters, seed=seed_value)
)

print("Iniciando el entrenamiento del modelo de clustering incremental (KMeans)...")

cluster_labels = [] # Almacena las etiquetas de clúster asignadas por el modelo

for x_instance in X_river_format:
    # 1. Predecir (Asignar la muestra al clúster más cercano)
    cluster_id = model_clustering.predict_one(x_instance)
    cluster_labels.append(cluster_id)
    
    # 2. Aprender (Actualizar el StandardScaler y los centroides de KMeans)
    model_clustering.learn_one(x_instance)

print(f"Entrenamiento completado. Muestras procesadas: {len(X_df)}\n")
cluster_labels = np.array(cluster_labels)

# --- 3. CORRECCIÓN DEL PCA Y VISUALIZACIÓN DE CLÚSTERES (2D) ---

## Paso 3.1: Aplicar el StandardScaler de River a todos los datos
river_scaler_instance = model_clustering['StandardScaler']

# Transformar TODAS las características usando el scaler entrenado
X_scaled_dicts = [river_scaler_instance.transform_one(row_dict) for row_dict in X_df.to_dict(orient='records')]
X_scaled = pd.DataFrame(X_scaled_dicts).values # Convertir a NumPy array para PCA

## Paso 3.2: Aplicar PCA a los datos ESCALADOS
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)

# Crear DataFrame para la visualización PCA
pca_df = pd.DataFrame(data=principal_components, columns=['PC 1', 'PC 2'])
pca_df['Cluster'] = cluster_labels 

## Paso 3.3: Generar la gráfica PCA
explained_variance_ratio_pc1 = pca.explained_variance_ratio_[0] * 100
explained_variance_ratio_pc2 = pca.explained_variance_ratio_[1] * 100

plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    pca_df['PC 1'], 
    pca_df['PC 2'], 
    c=pca_df['Cluster'],  
    cmap='viridis',       
    s=20,                 
    alpha=0.6
)

plt.title(f'Clustering KMeans (k={n_clusters}) en California Housing (SOLO CARACTERÍSTICAS)')
plt.xlabel(f'Componente Principal 1 ({explained_variance_ratio_pc1:.2f}% de Varianza)')
plt.ylabel(f'Componente Principal 2 ({explained_variance_ratio_pc2:.2f}% de Varianza)')

legend1 = plt.legend(*scatter.legend_elements(), title="Clústeres", loc="upper right")
plt.gca().add_artist(legend1)

plt.grid(True)
plt.show()

# --- 4. ANÁLISIS DE LOS CLUSTERS ENCONTRADOS ---

# Crear DataFrame de análisis con características y clusters
analysis_df = X_df.copy()
analysis_df['Cluster'] = cluster_labels
analysis_df['MedHouseVal'] = df[target_name].values  # Añadir target solo para análisis

# --- 5. VISUALIZACIÓN COMPARATIVA (BOX PLOT DE LA VARIABLE OBJETIVO POR CLUSTER) ---

plt.figure(figsize=(12, 6))

sns.boxplot(
    x='Cluster', 
    y=target_name, 
    data=analysis_df, 
    palette='viridis' 
)

plt.title(f'Distribución de {target_name} por Clúster (Clustering basado solo en características)')
plt.xlabel('ID del Clúster')
plt.ylabel('Valor Mediano de la Vivienda (Cientos de miles de $)')
plt.grid(axis='y', linestyle='--')

global_median = df[target_name].median()
plt.axhline(global_median, color='red', linestyle='dashed', linewidth=1, 
           label=f'Mediana Global: {global_median:.2f}')
plt.legend()

plt.show()

# --- 7. EVALUACIÓN DE LA CALIDAD DEL CLUSTERING ---

from sklearn.metrics import silhouette_score

# Calcular silhouette score (solo características)
silhouette_avg = silhouette_score(X_scaled, cluster_labels)
print(f"\n=== MÉTRICAS DE CALIDAD ===")
print(f"Silhouette Score: {silhouette_avg:.3f}")
print(f"Varianza explicada por PCA: {explained_variance_ratio_pc1 + explained_variance_ratio_pc2:.2f}%")
print(f"Número de muestras por cluster:")
print(analysis_df['Cluster'].value_counts().sort_index())