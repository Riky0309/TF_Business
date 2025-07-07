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





    

columnas_renombrar = {'Gender': 'Genero', 'Age': 'Edad', 'title': 'Titulo', 'country': 'Pais'}
df_netflix.rename(columns=columnas_renombrar, inplace=True)

# 2.2 Corrección de tipos de datos
columnas_castear = {'Genero': 'category', 'Titulo': 'category', 'Edad': 'category', 'Pais': 'category'}
df_netflix = df_netflix.astype(columnas_castear)

# 2.3 Eliminar columnas irrelevantes
df_netflix.drop(columns=['director', 'show_id', 'cast', 'description', 'Rational', 'date_added_month', 'date_added_day'], inplace=True)

# 2.4 Eliminar duplicados
df_netflix.drop_duplicates(inplace=True)

# 2.5 Manejo de Nulos
# Filtrado por filas con al menos 9 datos válidos
df_netflix.dropna(thresh=9, inplace=True)

# Eliminar columna 'Languages' por exceso de nulos
if 'Languages' in df_netflix.columns:
    df_netflix.drop(columns='Languages', inplace=True)

# Imputación con moda
for col in ['Genero', 'Pais']:
    if df_netflix[col].isnull().sum() > 0:
        df_netflix[col].fillna(df_netflix[col].mode()[0], inplace=True)

# Imputación de numéricos con mediana
num_cols = ['Cost Per Month - Premium ($)', 'Cost Per Month - Standard ($)']
for col in num_cols:
    if col in df_netflix.columns:
        df_netflix[col] = pd.to_numeric(df_netflix[col], errors='coerce')
        df_netflix[col].fillna(df_netflix[col].median(), inplace=True)

# ------------------------------
# 3️⃣ Análisis de Outliers
# ------------------------------
def plot_outliers(df, column_name):
    plt.figure(figsize=(6, 4))
    sns.boxplot(y=df[column_name])
    plt.title(f'Outliers en {column_name}')
    plt.grid(True)
    plt.show()

    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lim_inf = Q1 - 1.5 * IQR
    lim_sup = Q3 + 1.5 * IQR
    outliers = df[(df[column_name] < lim_inf) | (df[column_name] > lim_sup)]
    porcentaje = (len(outliers) / len(df)) * 100
    print(f"{column_name} - Outliers: {len(outliers)} ({porcentaje:.2f}%)")

for col in ['duration (min)', 'Cost Per Month - Premium ($)', 'No. of TV Shows']:
    if col in df_netflix.columns:
        df_netflix[col] = pd.to_numeric(df_netflix[col], errors='coerce')
        plot_outliers(df_netflix, col)

# ------------------------------
# 4️⃣ Transformación de Datos
# ------------------------------
from sklearn.preprocessing import LabelEncoder

# Codificar columnas categóricas con LabelEncoder
df_encoded = df_netflix.copy()
encoder = LabelEncoder()

mappings = {}
for col in ['Genero', 'Edad']:
    df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))
    mappings[col] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

# ------------------------------
# 5️⃣ Análisis Exploratorio (EDA) Visual
# ------------------------------

# Histograma de variables numéricas
numeric_columns = df_encoded.select_dtypes(include=['float64', 'int64']).columns
num_columns = len(numeric_columns)
num_rows = (num_columns // 5) + (num_columns % 5 > 0)

plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(num_rows, 5, i)
    sns.histplot(df_encoded[col], kde=True, color='skyblue', bins=20)
    plt.title(f'{col}')
    plt.tight_layout()
plt.show()

# ------------------------------
# 6️⃣ Insights Visuales Clave
# ------------------------------

# 1. Género del contenido vs Satisfacción
plt.figure(figsize=(10,6))
sns.boxplot(x='Genre_content', y='Satisfaction_score', data=df_netflix)
plt.title('🎭 Influencia del Género en la Satisfacción')
plt.xticks(rotation=90)
plt.show()

# 2. Duración vs Satisfacción
plt.figure(figsize=(10,6))
sns.scatterplot(x='duration (min)', y='Satisfaction_score', data=df_netflix)
plt.title('⏱️ Duración del Contenido vs Satisfacción')
plt.show()

# 3. Frecuencia de consumo vs Satisfacción
plt.figure(figsize=(10,6))
sns.barplot(x='Freq', y='Satisfaction_score', data=df_netflix, palette='Blues_d')
plt.title('🔁 Frecuencia de Consumo vs Satisfacción')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4. Género del usuario vs Satisfacción
plt.figure(figsize=(10,6))
sns.boxplot(x='Genero', y='Satisfaction_score', data=df_netflix)
plt.title('👤 Satisfacción según Género del Usuario')
plt.show()

# 5. Edad y satisfacción por género
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Edad', y='Satisfaction_score', hue='Genero', data=df_netflix, palette='Set2', alpha=0.7)
plt.title('🎯 Edad vs Satisfacción por Género')
plt.xlabel('Edad')
plt.ylabel('Satisfaction Score')
plt.show()

# 6. IMDB Score vs Satisfacción
if 'imdb_score' in df_netflix.columns:
    plt.figure(figsize=(10,6))
    sns.scatterplot(x='imdb_score', y='Satisfaction_score', data=df_netflix)
    plt.title('⭐ Calificación IMDB vs Satisfacción')
    plt.show()

# ------------------------------
# 🎉 Fin del Análisis Preliminar
# ------------------------------

print("✅ Análisis Exploratorio y Preprocesamiento completado.")










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

    cols_modelo = ['Age', 'Freq', 'duration (min)', 'Satisfaction_score']
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
