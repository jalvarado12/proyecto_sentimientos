import streamlit as st
import pandas as pd
import os
import glob
from datetime import datetime

# Intentar importar plotly con fallback
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly no disponible - usando gráficos básicos")

# Configuración de la página
st.set_page_config(
    page_title="Dashboard de Análisis de Sentimientos",
    layout="wide"
)

# Título principal
st.title("Dashboard de Análisis de Sentimientos en Tiempo Real")
st.markdown("---")

def load_local_sentiment_data():
    """Cargar datos desde archivos Parquet locales"""
    try:
        if not os.path.exists("stream_output"):
            return create_sample_data()
        
        parquet_files = glob.glob("stream_output/*.parquet") + glob.glob("stream_output/*.snappy.parquet")
        
        if not parquet_files:
            return create_sample_data()
        
        # Leer el primer archivo Parquet
        df = pd.read_parquet(parquet_files[0])
        st.success(f"Datos cargados: {len(df)} registros encontrados")
        return df
    
    except Exception as e:
        st.error(f"Error cargando datos: {str(e)}")
        return create_sample_data()

def create_sample_data():
    """Crear datos de ejemplo"""
    sample_data = {
        "tweet": [
            "Me encanta este producto, es increible!",
            "No me gusta para nada, muy decepcionado",
            "Excelente servicio al cliente",
            "Pesima calidad, no lo recomiendo", 
        ],
        "prediction_label": [1, 0, 1, 0],
        "positive_probability": [0.95, 0.15, 0.89, 0.23],
        "ingest_ts": [datetime.now()] * 4
    }
    return pd.DataFrame(sample_data)

# Cargar datos
with st.spinner("Cargando datos de analisis..."):
    df = load_local_sentiment_data()

# Calcular métricas
total_tweets = len(df)
positive_count = len(df[df['prediction_label'] == 1])
negative_count = len(df[df['prediction_label'] == 0])
positive_percentage = (positive_count / total_tweets) * 100 if total_tweets > 0 else 0

# Métricas principales
st.subheader("Metricas de Analisis en Tiempo Real")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Tweets", f"{total_tweets:,}")

with col2:
    st.metric("Positivos", f"{positive_count:,}", f"{positive_percentage:.1f}%")

with col3:
    st.metric("Negativos", f"{negative_count:,}", f"{(100-positive_percentage):.1f}%")

with col4:
    st.metric("Tasa Positividad", f"{positive_percentage:.1f}%")

# Visualizaciones
st.markdown("---")
st.subheader("Visualizaciones")

if PLOTLY_AVAILABLE:
    # Gráfico con Plotly
    sentiment_dist = pd.DataFrame({
        'Sentimiento': ['Positivo', 'Negativo'],
        'Cantidad': [positive_count, negative_count]
    })
    
    fig_pie = px.pie(
        sentiment_dist, 
        values='Cantidad', 
        names='Sentimiento',
        color='Sentimiento',
        color_discrete_map={'Positivo':'#00ff00', 'Negativo':'#ff0000'},
        title="Distribucion de Sentimientos"
    )
    st.plotly_chart(fig_pie, use_container_width=True)
else:
    # Gráfico básico sin Plotly
    st.write("**Distribucion de Sentimientos**")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Positivos:** {positive_count} ({positive_percentage:.1f}%)")
        st.progress(positive_percentage/100)
    with col2:
        st.write(f"**Negativos:** {negative_count} ({100-positive_percentage:.1f}%)")
        st.progress((100-positive_percentage)/100)

# Análisis recientes
st.markdown("---")
st.subheader("Analisis Recientes")

for idx, row in df.head(10).iterrows():
    sentiment_value = row['prediction_label']
    probability = row.get('positive_probability', 'N/A')
    
    if isinstance(probability, float):
        prob_formatted = f"{probability:.3f}"
    else:
        prob_formatted = str(probability)
    
    sentiment = "POSITIVO" if sentiment_value == 1 else "NEGATIVO"
    color = "#d4edda" if sentiment_value == 1 else "#f8d7da"
    border_color = "#c3e6cb" if sentiment_value == 1 else "#f5c6cb"
    text_color = "#155724" if sentiment_value == 1 else "#721c24"
    
    tweet_text = row['tweet']
    timestamp = row.get('ingest_ts', 'N/A')
    
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
st.subheader("Informacion Tecnica")

st.json({
    "total_registros": total_tweets,
    "positivos": positive_count,
    "negativos": negative_count,
    "porcentaje_positivo": f"{positive_percentage:.2f}%",
    "archivos_cargados": len(glob.glob("stream_output/*.parquet")) + len(glob.glob("stream_output/*.snappy.parquet"))
})

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6c757d;'>"
    "Proyecto Big Data - Analisis de Sentimientos con Spark ML<br>"
    "Dashboard desplegado en Streamlit Cloud"
    "</div>", 
    unsafe_allow_html=True
)
