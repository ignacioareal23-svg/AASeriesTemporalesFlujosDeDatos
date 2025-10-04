import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans # Importar KMeans

# --- 1. CARGA Y PREPARACIÓN DE DATOS ---
df = pd.read_csv('data/adult.csv')

# Definir la columna objetivo (income)
y_target = df['income']

# Eliminar la columna objetivo 'income' de las características
X = df.drop(columns=['income'])

# -------------------------------------------------------------
# --- 2. PRE-PROCESAMIENTO: ONE-HOT ENCODING (Manejo de Categóricas) ---
categorical_cols = X.select_dtypes(include=['object']).columns
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# -------------------------------------------------------------
# --- 3. ESCALADO DE DATOS ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Convertir el array escalado a lista de etiquetas para compatibilidad con la estructura anterior
X_scaled_array = X_scaled_df.values 

# -------------------------------------------------------------
# --- 4. CLUSTERING CON K-MEANS (K=2) ---

# Inicializar y entrenar el modelo KMeans con K=2
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10) # n_init=10 para asegurar convergencia
kmeans.fit(X_scaled_array)

# Obtener las etiquetas de cluster
cluster_labels = kmeans.labels_ # KMeans ya proporciona las etiquetas como un array

# -------------------------------------------------------------
# --- 5. PCA PARA VISUALIZACIÓN ---

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled_array)
X_pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
X_pca_df['Cluster'] = cluster_labels

# -------------------------------------------------------------
# --- 6. COMBINAR DATOS PARA MÉTRICAS Y GRÁFICOS ---

# Crear un DataFrame único con etiquetas de cluster y la variable objetivo
results_df = pd.DataFrame({
    'Cluster': cluster_labels,
    'Income': y_target
})

# -------------------------------------------------------------
# --- 7. IMPRESIÓN DE MÉTRICAS GENERALES ---

# 7.1. Número de Clusters (K=2, sin ruido)
n_clusters = 2
print(f"\n=======================================================")
print(f"|               MÉTRICAS DE CLUSTERING                |")
print(f"=======================================================")
print(f"| Número total de Clusters encontrados: {n_clusters:<15}|")
print(f"=======================================================")

# 7.2. Distribución de Instancias por Cluster
print("\nDistribución de Instancias por Cluster:")
cluster_distribution = results_df['Cluster'].value_counts().sort_index()
cluster_distribution.index = cluster_distribution.index.map(lambda x: f"Cluster {x}")
print(cluster_distribution.to_string())

# 7.3. Métrica de Silhouette Score
# K-Means siempre produce etiquetas, por lo que el cálculo es directo
if n_clusters > 1:
    silhouette_avg = silhouette_score(X_scaled_array, cluster_labels)
    print(f"\nSilhouette Score: {silhouette_avg:.4f}")
else:
    print("\nSilhouette Score: No se pudo calcular (solo un clúster).")

# -------------------------------------------------------------
# --- 8. GRÁFICO DE BARRAS (Distribución de 'income' por Clúster) ---

plot_df = results_df.copy()
plot_df['Cluster_Str'] = plot_df['Cluster'].astype(str)

plt.figure(figsize=(10, 6))
sns.countplot(
    data=plot_df, 
    x='Cluster_Str', 
    hue='Income', 
    palette='viridis'
)

# Configuración del gráfico de barras
plt.title('Distribución de Ingresos ("income") por Clúster (K-Means, K=2)')
plt.xlabel('ID del Clúster')
plt.ylabel('Conteo de Instancias')
plt.legend(title='Ingreso')
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()

# -------------------------------------------------------------
# --- 9. PLOTEO DE PCA ---

X_pca_df['Cluster_Str'] = X_pca_df['Cluster'].astype(str)

plt.figure(figsize=(12, 8))
sns.scatterplot(
    x='PC1',
    y='PC2',
    data=X_pca_df,
    hue='Cluster_Str',
    palette='Set1', # Usamos Set1 para dos colores claros
    s=25,
    alpha=0.7,
    legend='full'
)

# Configuración del gráfico
plt.title(f'K-Means Clustering (K=2) en Adult Dataset (PCA Reducido)')
plt.xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')

plt.legend(title='ID del Clúster')
plt.grid(True)
plt.tight_layout()
plt.show()