import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from river import cluster
from river import stream

# --- 1. CARGA Y PREPARACIÓN DE DATOS ---
data = fetch_california_housing(as_frame=True)
df = data.frame

# Obtener la columna objetivo antes de eliminarla para clustering/PCA
y_target = df['MedHouseVal']

# Eliminar la columna objetivo (NOTA: el nombre correcto es 'MedHouseVal')
X = df.drop(columns=['MedHouseVal'])

# -------------------------------------------------------------
# --- 2. PRE-PROCESAMIENTO PARA DBSTREAM (Escalado) ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Convertir a stream de diccionarios
X_dict_stream = X_scaled_df.to_dict('records')

# -------------------------------------------------------------
# --- 3. CLUSTERING INCREMENTAL CON DBSTREAM ---

# Parámetros utilizados en tu última ejecución
dbstream = cluster.DBSTREAM(fading_factor=0.01, clustering_threshold=4, intersection_factor=0.1)

# Aplicar el clustering incremental y recolectar etiquetas
cluster_labels = []
for x in X_dict_stream:
    dbstream.learn_one(x)
    label = dbstream.predict_one(x)
    # Asignar -1 para puntos no asignados/ruido
    cluster_labels.append(label if label is not None else -1)

# -------------------------------------------------------------
# --- 4. PCA PARA VISUALIZACIÓN ---

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
X_pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
X_pca_df['Cluster'] = cluster_labels

# -------------------------------------------------------------
# --- 5. COMBINAR DATOS PARA MÉTRICAS Y BOXPLOT ---

# Crear un DataFrame único con etiquetas de cluster y la variable objetivo
results_df = pd.DataFrame({
    'Cluster': cluster_labels,
    'MedHouseVal': y_target
})

# -------------------------------------------------------------
# --- 6. IMPRESIÓN DE MÉTRICAS GENERALES ---

# 6.1. Número de Clusters
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
print(f"\n=======================================================")
print(f"|               MÉTRICAS DE CLUSTERING                |")
print(f"=======================================================")
print(f"| Número total de Clusters encontrados: {n_clusters:<15}|")
print(f"=======================================================")

# 6.2. Distribución de Instancias por Cluster
print("\nDistribución de Instancias por Cluster:")
cluster_distribution = results_df['Cluster'].value_counts().sort_index()
cluster_distribution.index = cluster_distribution.index.map(lambda x: f"Cluster {x}" if x != -1 else "Ruido (-1)")
print(cluster_distribution.to_string())

# 6.3. Métrica de Silhouette Score
valid_indices = [i for i, label in enumerate(cluster_labels) if label != -1]

if len(valid_indices) > 1 and len(set(np.array(cluster_labels)[valid_indices])) > 1:
    X_valid = X_scaled[valid_indices]
    labels_valid = np.array(cluster_labels)[valid_indices]
    
    silhouette_avg = silhouette_score(X_valid, labels_valid)
    print(f"\nSilhouette Score (excluyendo Ruido): {silhouette_avg:.4f}")
else:
    print("\nSilhouette Score: No se pudo calcular (pocas instancias válidas o un solo cluster).")

# -------------------------------------------------------------
# --- 7. GRÁFICO DE CAJAS (Boxplot) ---

# Preparamos el DataFrame para el boxplot, etiquetando el ruido correctamente
plot_df = results_df.copy()
plot_df['Cluster_Str'] = plot_df['Cluster'].astype(str).replace('-1', 'Ruido')

plt.figure(figsize=(14, 6))
sns.boxplot(
    x='Cluster_Str', 
    y='MedHouseVal', 
    data=plot_df, 
    palette='Spectral' # Paleta de colores para buena distinción
)

# Configuración del gráfico de cajas
plt.title('Distribución de MedHouseVal por Clúster (DBSTREAM)')
plt.xlabel('ID del Clúster')
plt.ylabel('MedHouseVal (Valor Mediano de la Casa)')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()

# -------------------------------------------------------------
# --- 8. PLOTEO DE PCA (igual que antes) ---

X_pca_df['Cluster_Str'] = X_pca_df['Cluster'].astype(str).replace('-1', 'Ruido')

plt.figure(figsize=(12, 8))
sns.scatterplot(
    x='PC1',
    y='PC2',
    data=X_pca_df,
    hue='Cluster_Str',
    palette='tab20',
    s=25,
    alpha=0.7,
    legend='full'
)

# Configuración del gráfico
plt.title(f'DBSTREAM Clustering en California Housing (PCA Reducido)\n({n_clusters} clusters + Ruido)')
plt.xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')

handles, labels = plt.gca().get_legend_handles_labels()
labels = [l.replace('-1', 'Ruido') for l in labels]
plt.legend(handles, labels, title='ID del Cluster')

plt.grid(True)
plt.tight_layout()
plt.show()