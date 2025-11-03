# Archivo: preparar_datos_ejemplo.py
# Propósito: Crear archivos falsos (dummy) para que la app de Streamlit pueda ejecutarse.
# Ejecútalo una vez: python preparar_datos_ejemplo.py

import pandas as pd
import numpy as np
import os

print("Creando datos de ejemplo...")

NUM_MOVIES = 100
FEATURE_DIM = 64  # Dimensión de tus características (ej. 384)
NUM_CLUSTERS = 5
POSTER_FOLDER = 'posters'

# 1. Crear carpeta de pósters si no existe
if not os.path.exists(POSTER_FOLDER):
    os.makedirs(POSTER_FOLDER)
    print(f"Carpeta '{POSTER_FOLDER}' creada.")
    print(f"¡Añade algunos pósters (ej. p_0.jpg, p_1.jpg) allí para una mejor demo!")

# 2. Crear Metadatos (movies_data.csv)
genres_list = ['Acción', 'Comedia', 'Drama', 'Sci-Fi', 'Terror']
data = {
    'movie_id': [f'p_{i}' for i in range(NUM_MOVIES)],
    'title': [f'Película de {genres_list[i % 5]} #{i}' for i in range(NUM_MOVIES)],
    'genre': [genres_list[i % 5] for i in range(NUM_MOVIES)],
    'year': np.random.randint(1980, 2024, size=NUM_MOVIES),
    'poster_path': [f'{POSTER_FOLDER}/p_{i}.jpg' for i in range(NUM_MOVIES)]
}
df_movies = pd.DataFrame(data)
df_movies.to_csv('movies_data.csv', index=False)
print("Archivo 'movies_data.csv' creado.")

# 3. Crear Features (features.npy)
# Features aleatorias. Reemplaza esto con tu archivo .npy real.
features = np.random.rand(NUM_MOVIES, FEATURE_DIM)
np.save('features.npy', features)
print("Archivo 'features.npy' creado.")

# 4. Crear Features 2D (features_2d.npy)
# Simulación de PCA/t-SNE. Reemplaza con tu .npy real.
features_2d = np.random.rand(NUM_MOVIES, 2)
np.save('features_2d.npy', features_2d)
print("Archivo 'features_2d.npy' creado.")

# 5. Crear Etiquetas de Clúster (cluster_labels.npy)
# Asignaciones de clúster aleatorias. Reemplaza con tus etiquetas reales.
cluster_labels = np.random.randint(0, NUM_CLUSTERS, size=NUM_MOVIES)
np.save('cluster_labels.npy', cluster_labels)
print("Archivo 'cluster_labels.npy' creado.")

print("\n¡Completado! Ahora puedes ejecutar la app con:")
print("streamlit run app.py")