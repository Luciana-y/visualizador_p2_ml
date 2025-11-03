import pandas as pd
import requests

TMDB_API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyYWE2ZWRjNmFhY2I3YmRmYmY5ZmIzZTQwNTA1NzVhMiIsIm5iZiI6MTc2MTE2NzM3OS4xODksInN1YiI6IjY4Zjk0ODEzMmQ4MjliNDI0MzUzNjNmZiIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.8qNpyp-wtM85ZsTmrESFRwlizPY5cbe_Jyv5N2yef2Q"
def get_tmdb_poster(tmdb_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?language=en-US"
        headers = {"accept": "application/json", "Authorization": f"Bearer {TMDB_API_KEY}"}
        r = requests.get(url, headers=headers)
        if r.status_code == 200:
            data = r.json()
            poster_path = data.get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception as e:
        print(f"Error con {tmdb_id}: {e}")
    return None


# 1. Cargar dataset
df = pd.read_csv("unified_movies_clean.csv")

# 2. Generar columna nueva con URLs
df["poster_url"] = df["tmdbId"].apply(lambda x: get_tmdb_poster(int(x)) if pd.notnull(x) else None)

# 3. Guardar el resultado
df.to_csv("unified_movies_with_posters.csv", index=False)

print("✅ Listo. Archivo guardado como unified_movies_with_posters.csv")
