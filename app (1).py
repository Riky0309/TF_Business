
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Netflix Satisfaction Insights", layout="wide")

st.title("🎬 Análisis de Satisfacción de Usuarios de Netflix")

# Cargar archivo
uploaded_file = st.file_uploader("📁 Sube tu archivo de Excel (xlsx)", type="xlsx")

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.success("✅ Archivo cargado correctamente")

    # Mostrar columnas clave
    st.subheader("📊 Estructura de datos")
    st.write(df.head())

    # Insights
    st.header("🔍 Insights")

    # Insight 1
    if "Genre_content" in df.columns and "Satisfaction_score" in df.columns:
        st.subheader("1️⃣ Influencia del género en la satisfacción")
        fig, ax = plt.subplots(figsize=(10,6))
        sns.boxplot(x='Genre_content', y='Satisfaction_score', data=df, ax=ax)
        plt.xticks(rotation=90)
        st.pyplot(fig)
    else:
        st.warning("❗ Faltan columnas 'Genre_content' o 'Satisfaction_score'")

    # Insight 2
    if "duration (min)" in df.columns and "Satisfaction_score" in df.columns:
        st.subheader("2️⃣ Duración del contenido vs Satisfacción")
        fig, ax = plt.subplots(figsize=(10,6))
        sns.scatterplot(x='duration (min)', y='Satisfaction_score', data=df, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("❗ Faltan columnas 'duration (min)' o 'Satisfaction_score'")

    # Insight 3
    if "Freq" in df.columns and "Satisfaction_score" in df.columns:
        st.subheader("3️⃣ Frecuencia de uso y satisfacción")
        fig, ax = plt.subplots(figsize=(10,6))
        sns.barplot(x='Freq', y='Satisfaction_score', data=df, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.warning("❗ Faltan columnas 'Freq' o 'Satisfaction_score'")

    # Insight 4
    if "Genero" in df.columns and "Satisfaction_score" in df.columns:
        st.subheader("4️⃣ Satisfacción según el género del usuario")
        df_clean = df.dropna(subset=["Genero", "Satisfaction_score"]).copy()
        df_clean["Genero"] = df_clean["Genero"].astype(str)
        fig, ax = plt.subplots(figsize=(8,5))
        sns.boxplot(x='Genero', y='Satisfaction_score', data=df_clean, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("❗ Faltan columnas 'Genero' o 'Satisfaction_score'")


st.header("5️⃣ Insight: ¿Influencia de la edad en la satisfacción?")

if 'Edad' in df_netflix.columns and 'Satisfaction_score' in df_netflix.columns:
    df_temp = df_netflix[['Edad', 'Satisfaction_score']].dropna()
    df_temp['Edad'] = pd.to_numeric(df_temp['Edad'], errors='coerce')
    df_temp['Satisfaction_score'] = pd.to_numeric(df_temp['Satisfaction_score'], errors='coerce')
    df_temp = df_temp.dropna()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df_temp, x='Edad', y='Satisfaction_score', ax=ax)
    ax.set_title("Relación entre Edad y Nivel de Satisfacción")
    st.pyplot(fig)
else:
    st.warning("❗ Las columnas 'Edad' o 'Satisfaction_score' no están disponibles.")



else:
    st.info("🔄 Esperando que subas un archivo .xlsx válido.")







