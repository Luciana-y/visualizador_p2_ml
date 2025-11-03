# merge_csv_pkl.py
import pandas as pd
import pickle
import os

# =========================
# Rutas de archivos
# =========================
csv_path = "unified_movies_clean.csv"
pkl_pca_path = "pkl/j_clustered_movies_pca.pkl"
pkl_nmf_path = "pkl/j_clustered_movies_nmf.pkl"

output_dir = "merged_data"
os.makedirs(output_dir, exist_ok=True)

# =========================
# CARGAR CSV Y LIMPIAR tmdbId
# =========================
df = pd.read_csv(csv_path)

# Eliminar filas sin tmdbId
df = df.dropna(subset=["tmdbId"])
# Convertir de float a int
df["tmdbId"] = df["tmdbId"].astype(int)

# Usamos tmdbId como índice para el merge
df.set_index("tmdbId", inplace=True)

# =========================
# FUNCION PARA MERGEAR PKL
# =========================
def merge_with_cluster(df_csv, pkl_path, tipo="cluster"):
    # Cargar pkl
    with open(pkl_path, "rb") as f:
        cluster_df = pickle.load(f)
    
    # cluster_df debe tener tmdbId como índice
    cluster_df = cluster_df.copy()
    if cluster_df.index.name != "tmdbId":
        cluster_df.index.name = "tmdbId"
    
    # Mantener solo la columna "cluster" si tiene más
    if "cluster" not in cluster_df.columns:
        cluster_df = cluster_df.rename(columns={cluster_df.columns[0]: "cluster"})
    else:
        cluster_df = cluster_df[["cluster"]]
    
    # Merge
    df_merged = df_csv.merge(cluster_df, left_index=True, right_index=True, how="left")
    
    return df_merged

# =========================
# MERGE PCA
# =========================
df_merged_pca = merge_with_cluster(df, pkl_pca_path)
df_merged_pca.to_csv(os.path.join(output_dir, "movies_with_clusters_pca.csv"))
print("✅ Archivo mergeado PCA guardado en:", os.path.join(output_dir, "movies_with_clusters_pca.csv"))

# =========================
# MERGE NMF
# =========================
df_merged_nmf = merge_with_cluster(df, pkl_nmf_path)
df_merged_nmf.to_csv(os.path.join(output_dir, "movies_with_clusters_nmf.csv"))
print("✅ Archivo mergeado NMF guardado en:", os.path.join(output_dir, "movies_with_clusters_nmf.csv"))
