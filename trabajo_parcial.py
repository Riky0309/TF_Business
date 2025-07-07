import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import LabelEncoder

# ------------------------------
# Configuración de la Página
# ------------------------------
st.set_page_config(page_title="Netflix Satisfaction Insights", layout="wide")
st.title("🎬 Análisis de Satisfacción de Usuarios de Netflix")

# ------------------------------
# 1. 📁 Carga de Archivo
# ------------------------------
uploaded_file = st.file_uploader("📂 Sube tu archivo de Excel (.xlsx)", type="xlsx")

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.success("✅ Archivo cargado correctamente")

    # ------------------------------
    # 1️⃣ Vista Previa de los Datos Originales
    # ------------------------------
    st.subheader("📊 Vista Previa de los Datos Iniciales")
    st.markdown("A continuación se muestran las primeras 10 filas del archivo subido:")

    st.dataframe(df.head(10), use_container_width=True)

    # ------------------------------
    # 2.1 🧩 Combinación de Datos
    # ------------------------------
    columnas_renombrar = {
        'Gender': 'Genero',
        'Age': 'Edad',
        'title': 'Titulo',
        'country': 'Pais'
    }
    df_netflix = df.copy()
    df_netflix.rename(columns=columnas_renombrar, inplace=True)

    # ------------------------------
    # 2.2 🛠️ Corrección de Tipos de Datos
    # ------------------------------
    df_netflix['Titulo'] = df_netflix['Titulo'].astype(str)
    columnas_castear = {
        'Genero': 'category',
        'Titulo': 'category',
        'Edad': 'category',
        'Pais': 'category'
    }
    for col, tipo in columnas_castear.items():
        if col in df_netflix.columns:
            try:
                df_netflix[col] = df_netflix[col].astype(tipo)
            except Exception as e:
                st.warning(f"⚠️ No se pudo convertir la columna '{col}' a tipo {tipo}: {e}")

    # ------------------------------
    # 2.3 🧹 Eliminación de Columnas Irrelevantes
    # ------------------------------
    columnas_a_eliminar = [
        'director', 'show_id', 'cast', 'description',
        'Rational', 'date_added_month', 'date_added_day'
    ]
    df_netflix.drop(columns=[col for col in columnas_a_eliminar if col in df_netflix.columns], inplace=True)

    # ------------------------------
    # 2.4 🔁 Eliminación de Duplicados
    # ------------------------------
    df_netflix.drop_duplicates(inplace=True)

    # ------------------------------
    # 2.5 🧯 Filtrado de Filas con Muchos Nulos
    # ------------------------------
    df_netflix.dropna(thresh=9, inplace=True)

    # ------------------------------
    # 2.6 🩺 Manejo de Datos Faltantes (Missing Data)
    # ------------------------------
    if 'Languages' in df_netflix.columns:
        df_netflix.drop(columns='Languages', inplace=True)

    for col in ['Genero', 'Pais']:
        if col in df_netflix.columns and df_netflix[col].isnull().sum() > 0:
            df_netflix[col].fillna(df_netflix[col].mode()[0], inplace=True)

    num_cols = ['Cost Per Month - Premium ($)', 'Cost Per Month - Standard ($)']
    for col in num_cols:
        if col in df_netflix.columns:
            df_netflix[col] = pd.to_numeric(df_netflix[col], errors='coerce')
            df_netflix[col].fillna(df_netflix[col].median(), inplace=True)

    # ------------------------------
    # 3. 👁️ Vista Previa y Análisis Exploratorio
    # ------------------------------
    try:
        st.subheader("👁️ Vista previa de los primeros datos procesados")
        df_vista = df_netflix.select_dtypes(include=['number', 'category', 'object'])
        st.dataframe(df_vista.head(10))
    except Exception as e:
        st.error(f"❌ Error al mostrar la tabla: {e}")

    def plot_outliers(df, column_name):
        plt.figure(figsize=(6, 4))
        sns.boxplot(y=df[column_name])
        plt.title(f'Outliers en {column_name}')
        plt.grid(True)
        st.pyplot(plt)

    for col in ['duration (min)', 'Cost Per Month - Premium ($)', 'No. of TV Shows']:
        if col in df_netflix.columns:
            df_netflix[col] = pd.to_numeric(df_netflix[col], errors='coerce')
            plot_outliers(df_netflix, col)

    df_encoded = df_netflix.copy()
    encoder = LabelEncoder()
    mappings = {}
    for col in ['Genero', 'Edad']:
        df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))
        mappings[col] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

    numeric_columns = df_encoded.select_dtypes(include=['float64', 'int64']).columns
    num_columns = len(numeric_columns)
    num_rows = (num_columns // 5) + (num_columns % 5 > 0)

    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numeric_columns, 1):
        plt.subplot(num_rows, 5, i)
        sns.histplot(df_encoded[col], kde=True, color='skyblue', bins=20)
        plt.title(f'{col}')
        plt.tight_layout()
    st.pyplot(plt)

    # ------------------------------
    # 4. 🔎 Insights de Satisfacción
    # ------------------------------
    st.header("🔎 Insights")

    df_insight = df_netflix.copy()

    if "Genre_content" in df_insight.columns and "Satisfaction_score" in df_insight.columns:
        st.subheader("1️⃣ Influencia del género del contenido")
        fig, ax = plt.subplots(figsize=(10,6))
        sns.boxplot(x='Genre_content', y='Satisfaction_score', data=df_insight, ax=ax)
        plt.xticks(rotation=90)
        st.pyplot(fig)

    if "duration (min)" in df_insight.columns and "Satisfaction_score" in df_insight.columns:
        st.subheader("2️⃣ Duración vs Satisfacción")
        fig, ax = plt.subplots(figsize=(10,6))
        sns.scatterplot(x='duration (min)', y='Satisfaction_score', data=df_insight, ax=ax)
        st.pyplot(fig)

    if "Freq" in df_insight.columns and "Satisfaction_score" in df_insight.columns:
        st.subheader("3️⃣ Frecuencia de uso y satisfacción")
        fig, ax = plt.subplots(figsize=(10,6))
        sns.barplot(x='Freq', y='Satisfaction_score', data=df_insight, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    if "Genero" in df_insight.columns and "Satisfaction_score" in df_insight.columns:
        st.subheader("4️⃣ Satisfacción por género de usuario")
        df_clean = df_insight.dropna(subset=["Genero", "Satisfaction_score"]).copy()
        df_clean["Genero"] = df_clean["Genero"].astype(str)
        fig, ax = plt.subplots(figsize=(8,5))
        sns.boxplot(x='Genero', y='Satisfaction_score', data=df_clean, ax=ax)
        st.pyplot(fig)

    if "Edad" in df_insight.columns and "Satisfaction_score" in df_insight.columns:
        st.subheader("5️⃣ Satisfacción por edad")
        df_clean = df_insight.dropna(subset=["Edad", "Satisfaction_score"]).copy()
        df_clean["Edad"] = pd.to_numeric(df_clean["Edad"], errors="coerce")
        df_clean["Satisfaction_score"] = pd.to_numeric(df_clean["Satisfaction_score"], errors="coerce")
        df_clean.dropna(subset=["Edad", "Satisfaction_score"], inplace=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x="Edad", y="Satisfaction_score", data=df_clean, ax=ax)
        ax.set_title("Relación entre Edad y Satisfacción")
        st.pyplot(fig)

    # ------------------------------
    # 5. 🤖 Modelización con Random Forest
    # ------------------------------
    st.markdown("## 🤖 Modelización del Nivel de Satisfacción")
    st.markdown("Se entrena un modelo **Random Forest** para predecir la satisfacción del usuario.")

    cols_modelo = ['Edad', 'Freq', 'duration (min)', 'Satisfaction_score']
    if all(col in df_insight.columns for col in cols_modelo):
        with st.spinner("🧠 Entrenando el modelo..."):
            df_model = df_insight[cols_modelo].dropna()
            df_model['Freq'] = df_model['Freq'].astype('category').cat.codes
            df_model['Satisfaction_score'] = pd.cut(df_model['Satisfaction_score'],
                                                    bins=[0, 5, 8, 10],
                                                    labels=['Baja', 'Media', 'Alta'])

            X = df_model.drop('Satisfaction_score', axis=1)
            y = df_model['Satisfaction_score']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            modelo = RandomForestClassifier(n_estimators=100, random_state=42)
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)

        st.success("✅ Modelo entrenado correctamente.")

        st.markdown("### 🎯 Resultados del Modelo")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("📊 Accuracy", f"{accuracy_score(y_test, y_pred)*100:.2f}%")

        with col2:
            feature_importance = modelo.feature_importances_
            importance_df = pd.DataFrame({'Variable': X.columns, 'Importancia': feature_importance})
            fig, ax = plt.subplots()
            sns.barplot(x='Importancia', y='Variable', data=importance_df.sort_values(by="Importancia", ascending=True), ax=ax)
            ax.set_title("📌 Importancia de las Variables")
            st.pyplot(fig)

        st.markdown("### 📊 Matriz de Confusión")
        labels = unique_labels(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        cm_df = pd.DataFrame(
            cm,
            index=[f'Real {label}' for label in labels],
            columns=[f'Pred {label}' for label in labels]
        )
        st.dataframe(cm_df)

        st.markdown("### 📋 Reporte de Clasificación")
        st.code(classification_report(y_test, y_pred), language='text')

    else:
        st.error("❌ No se encontraron todas las columnas necesarias para entrenar el modelo.")
        st.write("Columnas requeridas:", cols_modelo)

else:
    st.warning("📂 Esperando que subas un archivo .xlsx válido.")

