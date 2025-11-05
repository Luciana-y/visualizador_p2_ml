import os
import pandas as pd

BASE_PATH_GMM = "../GMM_clusters"
BASE_PATH_KMEANS = "../KMEANSPP_Clusters"
MOVIES_PATH = "../unified_movies_clean.csv"

def load_clusters(algorithm="GMM", method="pca", dim="2d"):
    """Carga el CSV correspondiente al algoritmo, método y dimensión."""
    folder = BASE_PATH_GMM if algorithm == "GMM" else BASE_PATH_KMEANS
    file_prefix = "movies_clustered" if algorithm == "GMM" else "kmeanspp_clustered"
    filename = f"{file_prefix}_{method}_umap_{dim}.csv"
    path = os.path.join(folder, filename)
    return pd.read_csv(path)

def load_movies():
    """Carga el archivo base con la metadata de las películas."""
    df = pd.read_csv(MOVIES_PATH)
    df["movieId"] = df["movieId"].astype(str)
    df["tmdbId"] = pd.to_numeric(df["tmdbId"], errors="coerce")
    return df

def merge_movies_with_clusters(cluster_df, movies_df):
    """Une los datos del clustering con la metadata de películas y completa géneros si faltan."""
    cluster_df["movieId"] = cluster_df["movieId"].astype(str)
    merged = pd.merge(cluster_df, movies_df, on="movieId", how="left")

    # Si faltan columnas de géneros (caso KMEANSPP), las tomamos del movies_df
    genre_cols = [
        "Action","Adventure","Animation","Children","Comedy","Crime","Documentary",
        "Drama","Fantasy","Film-Noir","Horror","IMAX","Musical","Mystery",
        "Romance","Sci-Fi","Thriller","War","Western"
    ]

    for col in genre_cols:
        if col not in merged.columns:
            merged[col] = merged[col + "_y"] if (col + "_y") in merged.columns else movies_df.set_index("movieId")[col].reindex(merged["movieId"]).values

    # Si existen duplicados de columnas "_x" y "_y", limpiamos
    merged = merged.loc[:, ~merged.columns.duplicated()]

    return merged
