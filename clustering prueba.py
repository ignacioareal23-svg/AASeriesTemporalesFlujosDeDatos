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
n_clusters = 4
seed_value = 1

# Usaremos TODAS las columnas (incluyendo MedHouseVal) para el clustering.
X_df = df.copy() 
X_river_format = X_df.to_dict(orient='records') 

# --- 2. DEFINICIÓN Y ENTRENAMIENTO DEL PIPELINE (RIVER) ---

model_clustering = (
    preprocessing.StandardScaler() |
    cluster.KMeans(n_clusters=n_clusters, seed=seed_value)
)
# model_clustering = (
#     preprocessing.StandardScaler() |
#     cluster.DenStream(
#         beta=0.2,       
#         mu=8,           # Aumentado de 6 a 8 (mayor estrictez)
#         epsilon=0.02,   # Duplicado de 0.01 a 0.02 (mayor radio de clúster)
#     )
# )
print("Iniciando el entrenamiento del modelo de clustering incremental (KMeans)...")

cluster_labels = [] # Almacena las etiquetas de clúster asignadas por el modelo

for x_instance in X_river_format:
    # 1. Predecir (Asignar la muestra al clúster más cercano)
    cluster_id = model_clustering.predict_one(x_instance)
    cluster_labels.append(cluster_id)
    
    # 2. Aprender (Actualizar el StandardScaler y los centroides de KMeans)
    model_clustering.learn_one(x_instance)

print(f"Entrenamiento completado. Muestras procesadas: {len(df)}\n")
cluster_labels = np.array(cluster_labels)

# --- 3. CORRECCIÓN DEL PCA Y VISUALIZACIÓN DE CLÚSTERES (2D) ---

## Paso 3.1: Aplicar el StandardScaler de River a todos los datos (CORRECCIÓN CLAVE)
# ----------------------------------------------------------------------------------
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

plt.title(f'Clustering KMeans (k={n_clusters}) en California Housing (Proyección PCA ESCALADA)')
plt.xlabel(f'Componente Principal 1 ({explained_variance_ratio_pc1:.2f}% de Varianza)')
plt.ylabel(f'Componente Principal 2 ({explained_variance_ratio_pc2:.2f}% de Varianza)')

legend1 = plt.legend(*scatter.legend_elements(), title="Clústeres", loc="upper right")
plt.gca().add_artist(legend1)

plt.grid(True)
plt.show()

# --- 4. VISUALIZACIÓN COMPARATIVA (BOX PLOT DE LA VARIABLE OBJETIVO) ---

# Crear un DataFrame de comparación con la variable objetivo y las etiquetas de clúster
comparison_df = pd.DataFrame({
    target_name: df[target_name].values,
    'Cluster': cluster_labels
})
comparison_df['Cluster'] = comparison_df['Cluster'].astype('category')

# Generar la gráfica Box Plot
plt.figure(figsize=(12, 6))

sns.boxplot(
    x='Cluster', 
    y=target_name, 
    data=comparison_df, 
    palette='viridis' 
)

# Etiquetas y Referencia
plt.title(f'Distribución de {target_name} por Clúster')
plt.xlabel('ID del Clúster')
plt.ylabel('Valor Mediano de la Vivienda (Cientos de miles de $)')
plt.grid(axis='y', linestyle='--')

global_median = df[target_name].median()
plt.axhline(global_median, color='red', linestyle='dashed', linewidth=1, label=f'Mediana Global: {global_median:.2f}')
plt.legend()

plt.show()