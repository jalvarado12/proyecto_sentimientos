import streamlit as st
import pandas as pd
import plotly.express as px
import os
import glob
from datetime import datetime
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

# Configuración de la página
st.set_page_config(
    page_title="Dashboard de Análisis de Sentimientos",
    page_icon="📊",
    layout="wide"
)

# Título principal
st.title("Dashboard de Análisis de Sentimientos en Tiempo Real")
st.markdown("---")

def load_local_sentiment_data():
    """Cargar datos desde archivos Parquet locales eliminando duplicados"""
    try:
        if not os.path.exists("stream_output"):
            return create_sample_data()
        
        parquet_files = glob.glob("stream_output/*.parquet") + glob.glob("stream_output/*.snappy.parquet")
        
        if not parquet_files:
            return create_sample_data()
        
        # Leer todos los archivos pero eliminar duplicados
        dfs = []
        for file in parquet_files:
            try:
                df = pd.read_parquet(file)
                if not df.empty:
                    dfs.append(df)
            except Exception:
                continue
        
        if not dfs:
            return create_sample_data()
        
        # Combinar todos los DataFrames
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Eliminar duplicados basados en columnas clave
        if 'tweet' in combined_df.columns and 'ingest_ts' in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=['tweet', 'ingest_ts'])
        elif 'tweet' in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=['tweet'])
        else:
            combined_df = combined_df.drop_duplicates()
        
        return combined_df
    
    except Exception:
        return create_sample_data()

def create_sample_data():
    """Crear datos de ejemplo más robustos"""
    sample_data = {
        "tweet": [
            "Me encanta este producto, es increíble! ",
            "No me gusta para nada, muy decepcionado ",
            "Excelente servicio al cliente, muy profesionales ",
            "Pésima calidad, no lo recomiendo ",
            "Increíble experiencia, volvería a comprar sin duda ",
            "Muy mala atención, no responden las consultas ",
            "Producto de buena calidad pero delivery lento ",
            "Totalmente satisfecho con mi compra ",
            "No cumple con lo prometido, muy decepcionante ",
            "Rápido y eficiente, excelente trabajo "
        ],
        "prediction_label": [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
        "positive_probability": [0.95, 0.15, 0.89, 0.23, 0.92, 0.18, 0.76, 0.88, 0.27, 0.91],
        "ingest_ts": [datetime.now() for _ in range(10)]
    }
    return pd.DataFrame(sample_data)

def analyze_sentiment_data(df):
    """Analizar datos de sentimientos con la estructura correcta"""
    
    # Usar las columnas reales de tu dataset
    text_col = 'tweet'
    prediction_col = 'prediction_label'
    probability_col = 'positive_probability'
    timestamp_col = 'ingest_ts'
    
    # Calcular métricas
    total_tweets = len(df)
    positive_count = len(df[df[prediction_col] == 1]) if prediction_col in df.columns else 0
    negative_count = len(df[df[prediction_col] == 0]) if prediction_col in df.columns else 0
    positive_percentage = (positive_count / total_tweets) * 100 if total_tweets > 0 else 0
    
    # Calcular probabilidad promedio
    avg_positive_prob = df[probability_col].mean() if probability_col in df.columns else 0
    
    return {
        'total_tweets': total_tweets,
        'positive_count': positive_count,
        'negative_count': negative_count,
        'positive_percentage': positive_percentage,
        'avg_positive_prob': avg_positive_prob,
        'text_col': text_col,
        'prediction_col': prediction_col,
        'probability_col': probability_col,
        'timestamp_col': timestamp_col,
        'df': df
    }

def generate_wordcloud(df, sentiment_type):
    """Generar nube de palabras para sentimientos positivos o negativos"""
    try:
        # Filtrar datos por sentimiento
        if sentiment_type == 'positive':
            filtered_df = df[df['prediction_label'] == 1]
            title = "Palabras Más Frecuentes - Sentimientos Positivos"
            colormap = 'Greens'
        else:
            filtered_df = df[df['prediction_label'] == 0]
            title = "Palabras Más Frecuentes - Sentimientos Negativos"
            colormap = 'Reds'
        
        if len(filtered_df) == 0:
            return None
        
        # Combinar todo el texto
        text = ' '.join(filtered_df['tweet'].astype(str))
        
        if len(text.strip()) == 0:
            return None
        
        # Crear word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap=colormap,
            max_words=50,
            contour_width=1,
            contour_color='steelblue',
            stopwords=['que', 'de', 'en', 'y', 'la', 'el', 'los', 'las', 'un', 'una', 'es', 'se', 'no']
        ).generate(text)
        
        # Crear figura
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        return fig
        
    except Exception:
        return None

# Sidebar
st.sidebar.title("Configuración")
if st.sidebar.button(" Actualizar Datos"):
    st.rerun()

# Cargar y analizar datos
with st.spinner("Cargando datos de análisis..."):
    df = load_local_sentiment_data()
    analysis = analyze_sentiment_data(df)

# Extraer resultados
total_tweets = analysis['total_tweets']
positive_count = analysis['positive_count']
negative_count = analysis['negative_count']
positive_percentage = analysis['positive_percentage']
avg_positive_prob = analysis['avg_positive_prob']
text_col = analysis['text_col']
prediction_col = analysis['prediction_col']
probability_col = analysis['probability_col']
timestamp_col = analysis['timestamp_col']
df = analysis['df']

# Métricas principales
st.subheader("Métricas de Análisis en Tiempo Real")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Tweets", f"{total_tweets:,}")

with col2:
    st.metric("Positivos", f"{positive_count:,}", f"{positive_percentage:.1f}%")

with col3:
    st.metric("Negativos", f"{negative_count:,}", f"{(100-positive_percentage):.1f}%")

with col4:
    st.metric("Tasa Positividad", f"{positive_percentage:.1f}%")

with col5:
    st.metric("Prob. Positiva Avg", f"{avg_positive_prob:.2f}")

# Gráficos principales
st.markdown("---")
st.subheader("Visualizaciones Principales")

col1, col2 = st.columns(2)

with col1:
    # Distribución de sentimientos
    sentiment_dist = pd.DataFrame({
        'Sentimiento': ['Positivo', 'Negativo'],
        'Cantidad': [positive_count, negative_count]
    })
    
    fig_pie = px.pie(
        sentiment_dist, 
        values='Cantidad', 
        names='Sentimiento',
        color='Sentimiento',
        color_discrete_map={'Positivo':'#2ecc71', 'Negativo':'#e74c3c'},
        title="Distribución de Sentimientos"
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    # Gráfico de probabilidades si hay datos suficientes
    if total_tweets > 1 and probability_col in df.columns:
        fig_hist = px.histogram(
            df, 
            x=probability_col,
            title="Distribución de Probabilidades Positivas",
            labels={probability_col: 'Probabilidad de Sentimiento Positivo'},
            color_discrete_sequence=['#3498db']
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        fig_bar = px.bar(
            sentiment_dist,
            x='Sentimiento',
            y='Cantidad',
            color='Sentimiento',
            color_discrete_map={'Positivo':'#2ecc71', 'Negativo':'#e74c3c'},
            title="Conteo por Sentimiento"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# Word Clouds
st.markdown("---")
st.subheader("Análisis de Texto - Nubes de Palabras")

if total_tweets > 0:
    col1, col2 = st.columns(2)
    
    with col1:
        # Word Cloud para sentimientos positivos
        wordcloud_pos = generate_wordcloud(df, 'positive')
        if wordcloud_pos:
            st.pyplot(wordcloud_pos)
        else:
            st.info("No hay datos suficientes para generar nube de palabras positivas")
    
    with col2:
        # Word Cloud para sentimientos negativos
        wordcloud_neg = generate_wordcloud(df, 'negative')
        if wordcloud_neg:
            st.pyplot(wordcloud_neg)
        else:
            st.info("No hay datos suficientes para generar nube de palabras negativas")
else:
    st.info("No hay datos suficientes para generar nubes de palabras")

# Análisis recientes
st.markdown("---")
st.subheader("Análisis Recientes")

# Ordenar por timestamp si está disponible
if timestamp_col in df.columns and timestamp_col in df.columns:
    recent_data = df.sort_values(timestamp_col, ascending=False).head(10)
else:
    recent_data = df.head(10)

for idx, row in recent_data.iterrows():
    sentiment_value = row[prediction_col] if prediction_col in df.columns else 1
    probability = row.get(probability_col, 'N/A')
    
    # Formatear la probabilidad correctamente
    if isinstance(probability, (int, float)):
        prob_formatted = f"{probability:.3f}"
    else:
        prob_formatted = str(probability)
    
    sentiment = "POSITIVO" if sentiment_value == 1 else "NEGATIVO"
    color = "#d4edda" if sentiment_value == 1 else "#f8d7da"
    border_color = "#c3e6cb" if sentiment_value == 1 else "#f5c6cb"
    text_color = "#155724" if sentiment_value == 1 else "#721c24"
    
    tweet_text = row[text_col] if text_col in df.columns else "Texto no disponible"
    timestamp = row.get(timestamp_col, 'N/A')
    
    st.markdown(f"""
    <div style='background-color: {color}; border: 2px solid {border_color}; 
                border-radius: 8px; padding: 12px; margin: 8px 0;'>
        <div style='font-weight: bold; color: {text_color}; font-size: 14px;'>
            {sentiment} - Probabilidad: {prob_formatted}
        </div>
        <div style='margin: 8px 0; font-size: 13px; line-height: 1.4;'>{tweet_text}</div>
        <div style='font-size: 11px; color: #6c757d;'>
            Timestamp: {timestamp}
        </div>
    </div>
    """, unsafe_allow_html=True)

# Información técnica
st.markdown("---")
st.subheader("Información Técnica")

col1, col2 = st.columns(2)

with col1:
    st.write("**Estructura del Dataset:**")
    st.json({
        "total_registros": total_tweets,
        "positivos": positive_count,
        "negativos": negative_count,
        "porcentaje_positivo": f"{positive_percentage:.2f}%",
        "probabilidad_promedio": f"{avg_positive_prob:.3f}"
    })

with col2:
    st.write("**Columnas Disponibles:**")
    available_cols = df.columns.tolist()
    for col in available_cols:
        st.write(f"- `{col}`")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6c757d;'>"
    "Proyecto Big Data - Análisis de Sentimientos con Spark ML"
    "</div>", 
    unsafe_allow_html=True
)
