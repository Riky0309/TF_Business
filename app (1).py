import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels  # ✅ Importación necesaria

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
    if "Gender" in df.columns and "Satisfaction_score" in df.columns:
        st.subheader("4️⃣ Satisfacción según el género del usuario")
        df_clean = df.dropna(subset=["Gender", "Satisfaction_score"]).copy()
        df_clean["Gender"] = df_clean["Gender"].astype(str)
        fig, ax = plt.subplots(figsize=(8,5))
        sns.boxplot(x='Gender', y='Satisfaction_score', data=df_clean, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("❗ Faltan columnas 'Gender' o 'Satisfaction_score'")

    # Insight 5
    if "Age" in df.columns and "Satisfaction_score" in df.columns:
        st.subheader("5️⃣ Influencia de la edad en la satisfacción del usuario")
        df_clean = df.dropna(subset=["Age", "Satisfaction_score"]).copy()
        df_clean["Age"] = pd.to_numeric(df_clean["Age"], errors="coerce")
        df_clean["Satisfaction_score"] = pd.to_numeric(df_clean["Satisfaction_score"], errors="coerce")
        df_clean = df_clean.dropna(subset=["Age", "Satisfaction_score"])
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x="Age", y="Satisfaction_score", data=df_clean, ax=ax)
        ax.set_title("Relación entre Edad y Nivel de Satisfacción")
        st.pyplot(fig)
    else:
        st.warning("❗ Faltan columnas 'Age' o 'Satisfaction_score'")

    # Modelado
    st.markdown("## 📈 4. Modelización del Nivel de Satisfacción")
    st.markdown("A continuación se entrena un modelo **Random Forest** para predecir el nivel de satisfacción del usuario en función de su edad, frecuencia de uso y duración de contenido.")

    cols_modelo = ['Edad', 'Freq', 'duration (min)', 'Satisfaction_score']
    if all(col in df.columns for col in cols_modelo):
        with st.spinner("🔄 Procesando datos y entrenando el modelo..."):
            df_model = df[cols_modelo].dropna()
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
            ax.set_title("📌 Importancia de las variables")
            st.pyplot(fig)

        st.markdown("### 📋 Matriz de Confusión")

        labels = unique_labels(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=labels)

        cm_df = pd.DataFrame(
            cm,
            index=[f'Real {label}' for label in labels],
            columns=[f'Pred {label}' for label in labels]
        )

        st.dataframe(cm_df)

        st.markdown("### 📄 Reporte de Clasificación")
        st.code(classification_report(y_test, y_pred), language='text')

    else:
        st.error("🚫 No se encontraron todas las columnas necesarias para el modelo:")
        st.write("Se requieren:", cols_modelo)

else:
    st.warning("🔄 Esperando que subas un archivo .xlsx válido.")
