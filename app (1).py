
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

# Insight 5
    if "Edad" in df.columns and "Satisfaction_score" in df.columns:
        st.subheader("5️⃣ Influencia de la edad en la satisfacción del usuario")
        df_clean = df.dropna(subset=["Edad", "Satisfaction_score"]).copy()
        df_clean["Edad"] = pd.to_numeric(df_clean["Edad"], errors="coerce")
        df_clean["Satisfaction_score"] = pd.to_numeric(df_clean["Satisfaction_score"], errors="coerce")
        df_clean = df_clean.dropna(subset=["Edad", "Satisfaction_score"])

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x="Edad", y="Satisfaction_score", data=df_clean, ax=ax)
        ax.set_title("Relación entre Edad y Nivel de Satisfacción")
        st.pyplot(fig)
    else:
        st.warning("❗ Faltan columnas 'Edad' o 'Satisfaction_score'")


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.header("📈 4. Modelización")

# Verifica que estén todas las columnas necesarias
cols_modelo = ['Edad', 'Freq', 'duration (min)', 'Satisfaction_score']
if all(col in df_netflix.columns for col in cols_modelo):
    df_model = df_netflix[cols_modelo].dropna()

    # Conversión de variables categóricas
    df_model['Freq'] = df_model['Freq'].astype('category').cat.codes
    df_model['Satisfaction_score'] = pd.cut(df_model['Satisfaction_score'],
                                            bins=[0, 5, 8, 10],
                                            labels=['Baja', 'Media', 'Alta'])

    # Separación de variables
    X = df_model.drop('Satisfaction_score', axis=1)
    y = df_model['Satisfaction_score']

    # División en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Entrenamiento del modelo
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    # Resultados
    st.subheader("🔍 Resultados del modelo Random Forest")
    st.write("**Accuracy:**", round(accuracy_score(y_test, y_pred), 2))

    st.text("Matriz de Confusión:")
    st.write(confusion_matrix(y_test, y_pred))

    st.text("Reporte de Clasificación:")
    st.text(classification_report(y_test, y_pred))

else:
    st.warning("⚠️ Faltan columnas necesarias para el modelo: " + ', '.join(cols_modelo))

else:
    st.warning("🔄 Esperando que subas un archivo .xlsx válido.")







