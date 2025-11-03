import streamlit as st
import pandas as pd
import re

st.set_page_config(layout="wide")
st.title("🎬 Sistema de Recomendación de Películas")
st.header("Bloque 1: Filtros por Metadatos")

GENRE_LIST = [
    'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 
    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 
    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]

@st.cache_data
def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        
        df['year_str'] = df['title'].str.extract(r'\((\d{4})\)$')
        
        df['year'] = pd.to_numeric(df['year_str'], errors='coerce')
        
        df = df.dropna(subset=['year'])
        
        df['year'] = df['year'].astype(int)
        
        return df
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo en {filepath}. Asegúrate de que esté en el directorio correcto.")
        return pd.DataFrame()

df_movies = load_data('movies_train.csv')

if not df_movies.empty:
    st.sidebar.header("Filtros de Películas")
    
    selected_genres = st.sidebar.multiselect(
        "Selecciona géneros:",
        options=GENRE_LIST,
        default=[]
    )
    
    min_year = df_movies['year'].min()
    max_year = df_movies['year'].max()
    
    year_range = st.sidebar.slider(
        "Selecciona un rango de años:",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )

    df_filtered = df_movies.copy()

    if selected_genres:
        for genre in selected_genres:
            if genre in df_filtered.columns:
                df_filtered = df_filtered[df_filtered[genre] == 1]

    df_filtered = df_filtered[
        (df_filtered['year'] >= year_range[0]) & 
        (df_filtered['year'] <= year_range[1])
    ]

    st.subheader(f"Mostrando {len(df_filtered)} de {len(df_movies)} películas")
    
    st.dataframe(df_filtered[['title', 'genres', 'year']])

else:
    st.warning("No se pudieron cargar los datos de las películas para iniciar la aplicación.")