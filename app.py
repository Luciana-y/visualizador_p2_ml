import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from PIL import Image
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from skimage.feature import local_binary_pattern, hog

# ============================
# FUNCIÓN DE EXTRACCIÓN DE CARACTERÍSTICAS VISUALES
# ============================
def extract_features(image):
    try:
        # Convertir PIL.Image a array compatible con OpenCV
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        image = cv2.resize(image, (256, 256))

        # --- Histograma HSV ---
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist_hsv = cv2.calcHist([hsv_image], [0, 1, 2], None, [8, 8, 4],
                                [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist_hsv, hist_hsv)
        hist_hsv = hist_hsv.flatten()

        # --- LBP ---
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray_image, P=24, R=8, method="uniform")
        (hist_lbp, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
        hist_lbp = hist_lbp.astype("float")
        hist_lbp /= (hist_lbp.sum() + 1e-6)

        # --- HOG ---
        gray_image = cv2.resize(gray_image, (128, 128))
        features_hog = hog(
            gray_image,
            orientations=6,
            pixels_per_cell=(16, 16),
            cells_per_block=(3, 3),
            transform_sqrt=True,
            block_norm="L2-Hys"
        )

        # Normalización
        hist_hsv /= np.linalg.norm(hist_hsv) + 1e-6
        hist_lbp /= np.linalg.norm(hist_lbp) + 1e-6
        features_hog /= np.linalg.norm(features_hog) + 1e-6

        return np.hstack([hist_hsv, hist_lbp, features_hog])

    except Exception as e:
        print(f"Error procesando imagen: {e}")
        return None

# ============================
# CONFIGURACIÓN INICIAL
# ============================
st.set_page_config(page_title="Visual Similarity Explorer", layout="wide")
TMDB_API_KEY = "TU_API_KEY"  # Reemplazar con tu API Key de TMDB
st.title("🎞️ Visual Similarity Movie Explorer")

# ============================
# BLOQUE 1: SUBIR EMBEDDINGS
# ============================
st.sidebar.header("1️⃣ Subir archivo de embeddings (.pkl)")
uploaded_file = st.sidebar.file_uploader("Selecciona el archivo .pkl con los embeddings", type=["pkl"])

st.sidebar.header("1.1️⃣ (Opcional) Subir modelo KMeans++")
uploaded_model = st.sidebar.file_uploader("Sube tu modelo KMeans++ (.pkl)", type=["pkl"])

kmeans_model = None
if uploaded_model is not None:
    try:
        kmeans_model = pickle.load(uploaded_model)
        st.sidebar.success("✅ Modelo KMeans cargado correctamente")
    except Exception as e:
        st.sidebar.error(f"Error al cargar el modelo: {e}")

if uploaded_file is not None:
    try:
        data = pickle.load(uploaded_file)
        st.sidebar.success("✅ Archivo cargado correctamente")

        # Puede ser dict o dataframe
        if isinstance(data, dict):
            movie_ids = list(data.keys())
            embeddings = np.array(list(data.values()))
        elif isinstance(data, pd.DataFrame):
            movie_ids = data["movieId"].tolist()
            embeddings = np.vstack([np.array(e) for e in data["embedding"].values])
        else:
            st.error("Formato de archivo no reconocido. Usa dict o DataFrame con columnas ['movieId', 'embedding']")
            st.stop()
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        st.stop()
else:
    st.warning("Sube primero tu archivo .pkl con los embeddings para continuar.")
    st.stop()

# ============================
# BLOQUE 2: SUBIR IMAGEN DE CONSULTA
# ============================
st.sidebar.header("2️⃣ Subir imagen de referencia")
query_img_file = st.sidebar.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])

if query_img_file:
    query_img = Image.open(query_img_file).convert("RGB")
    st.image(query_img, caption="Imagen de consulta", width=200)

    # Extraer características reales de la imagen subida
    query_vector = extract_features(query_img)
    if query_vector is None:
        st.error("No se pudieron extraer características de la imagen.")
        st.stop()
    else:
        query_vector = query_vector.reshape(1, -1)

        # ============================
        # BLOQUE 3: CÁLCULO DE SIMILITUD
        # ============================
        sims = cosine_similarity(query_vector, embeddings)[0]
        top_idx = np.argsort(sims)[::-1][:5]
        top_movies = [movie_ids[i] for i in top_idx]
        top_scores = [sims[i] for i in top_idx]

        st.subheader("🎬 Películas más similares visualmente")
        cols = st.columns(5)
        for i, (movie_id, score) in enumerate(zip(top_movies, top_scores)):
            # Consultar TMDB para obtener poster
            url = f"https://api.themoviedb.org/3/movie/{movie_id}?language=en-US"
            headers = {"accept": "application/json", "Authorization": f"Bearer {TMDB_API_KEY}"}
            r = requests.get(url, headers=headers)
            if r.status_code == 200:
                data_tmdb = r.json()
                poster_path = data_tmdb.get("poster_path")
                title = data_tmdb.get("title", "Unknown")
                if poster_path:
                    poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                    response = requests.get(poster_url)
                    img = Image.open(BytesIO(response.content))
                    cols[i].image(img, caption=f"{title}\n({score:.2f})")
                else:
                    cols[i].write(f"{title}\n({score:.2f})")
        # Mostrar cluster si modelo KMeans está cargado
        if kmeans_model is not None:
            cluster_id = kmeans_model.predict(query_vector)[0]
            st.info(f"📊 La imagen pertenece al cluster: **{cluster_id}**")
else:
    st.info("Sube una imagen para ver las películas similares.")
