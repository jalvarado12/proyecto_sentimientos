import streamlit as st
import os
import glob
from datetime import datetime

# Configuración de la página
st.set_page_config(
    page_title="Dashboard de Analisis de Sentimientos",
    layout="wide"
)

# Título principal
st.title("Dashboard de Analisis de Sentimientos en Tiempo Real")
st.markdown("---")

# Intentar importar TODAS las dependencias con fallbacks robustos
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    st.error("Pandas no disponible - Funcionalidad limitada")

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly no disponible - Usando graficos basicos")

try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    st.warning("WordCloud no disponible - Analisis de texto limitado")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    st.warning("NumPy no disponible - Calculos limitados")

def create_sample_data():
    """Crear datos de ejemplo robustos"""
    sample_data = {
        "tweet": [
            "Me encanta este producto, es increible la calidad que ofrece",
            "No me gusta para nada, muy decepcionado con el servicio",
            "Excelente atencion al cliente, resolvieron mi problema rapido",
            "Pesima calidad, el producto se dano en la primera semana",
            "Entrega rapida y producto en perfecto estado, recomendado",
            "Mala experiencia, el soporte tecnico no sabe resolver",
            "Funcionalidades avanzadas a un precio muy competitivo",
            "Interfaz confusa y dificil de usar para principiantes",
            "Actualizaciones constantes que mejoran la experiencia",
            "Problemas de compatibilidad con otros dispositivos",
            "Diseno moderno y materiales de alta calidad",
            "Documentacion incompleta y poco clara",
            "Rendimiento excelente en todas las pruebas",
            "Consume demasiada bateria y recursos del sistema",
            "Soporte multilenguaje muy bien implementado",
            "Faltan caracteristicas basicas esperadas",
            "Estabilidad notable incluso bajo carga pesada",
            "Frecuentes caidas y reinicios inesperados",
            "Comunidad activa y soporte excelente",
            "Poco mantenimiento y actualizaciones escasas"
        ],
        "prediction_label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        "positive_probability": [
            0.95, 0.12, 0.89, 0.18, 0.92, 0.15, 0.88, 0.22, 0.91, 0.19,
            0.93, 0.16, 0.94, 0.17, 0.90, 0.21, 0.87, 0.14, 0.96, 0.13
        ],
        "ingest_ts": [datetime.now()] * 20
    }
    return sample_data

def load_sentiment_data():
    """Cargar datos con manejo robusto de errores"""
    if not PANDAS_AVAILABLE:
        st.error("No se pueden cargar datos sin pandas")
        return create_sample_data()
    
    try:
        # Intentar cargar datos reales
        if os.path.exists("stream_output"):
            parquet_files = glob.glob("stream_output/*.parquet") + glob.glob("stream_output/*.snappy.parquet")
            if parquet_files:
                df = pd.read_parquet(parquet_files[0])
                st.success(f"Datos reales cargados: {len(df)} registros")
                return df
        
        # Datos de ejemplo como fallback
        sample_data = create_sample_data()
        df = pd.DataFrame(sample_data)
        st.info("Usando datos de ejemplo demostrativos")
        return df
        
    except Exception as e:
        st.error(f"Error cargando datos: {str(e)}")
        sample_data = create_sample_data()
        return pd.DataFrame(sample_data)

def generate_wordcloud_safe(text_data, title, color_scheme='viridis'):
    """Generar wordcloud con manejo de errores"""
    if not WORDCLOUD_AVAILABLE or not text_data:
        return None
    
    try:
        # Combinar texto
        text = ' '.join(str(tweet) for tweet in text_data if tweet)
        
        if not text.strip():
            return None
        
        # Crear wordcloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap=color_scheme,
            max_words=50,
            contour_width=1,
            contour_color='steelblue'
        ).generate(text)
        
        # Crear figura
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        st.error(f"Error generando wordcloud: {str(e)}")
        return None

# Sidebar para controles
st.sidebar.title("Configuracion")
if st.sidebar.button("Actualizar Datos"):
    st.rerun()

# Cargar datos
if PANDAS_AVAILABLE:
    with st.spinner("Cargando datos de analisis..."):
        df = load_sentiment_data()
    
    # Calcular metricas
    total_tweets = len(df)
    positive_count = len(df[df['prediction_label'] == 1])
    negative_count = len(df[df['prediction_label'] == 0])
    positive_percentage = (positive_count / total_tweets) * 100 if total_tweets > 0 else 0
    
    # Metricas principales
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
        # Grafico Plotly
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
        # Graficos basicos
        col1, col2 = st.columns(2)
        with col1:
            st.write("Distribucion de Sentimientos")
            st.write(f"Positivos: {positive_count} ({positive_percentage:.1f}%)")
            st.progress(positive_percentage/100)
        with col2:
            st.write(f"Negativos: {negative_count} ({100-positive_percentage:.1f}%)")
            st.progress((100-positive_percentage)/100)
    
    # WORD CLOUDS - ESENCIAL
    st.markdown("---")
    st.subheader("Analisis de Texto - Nubes de Palabras")
    
    if WORDCLOUD_AVAILABLE:
        col1, col2 = st.columns(2)
        
        with col1:
            # Word Cloud Positivo
            positive_tweets = df[df['prediction_label'] == 1]['tweet'].tolist()
            wc_pos = generate_wordcloud_safe(positive_tweets, "Palabras Mas Frecuentes - Sentimientos Positivos", 'Greens')
            if wc_pos:
                st.pyplot(wc_pos)
            else:
                st.info("No hay datos suficientes para wordcloud positivo")
        
        with col2:
            # Word Cloud Negativo
            negative_tweets = df[df['prediction_label'] == 0]['tweet'].tolist()
            wc_neg = generate_wordcloud_safe(negative_tweets, "Palabras Mas Frecuentes - Sentimientos Negativos", 'Reds')
            if wc_neg:
                st.pyplot(wc_neg)
            else:
                st.info("No hay datos suficientes para wordcloud negativo")
    else:
        st.error("WordCloud no disponible - Instala wordcloud y matplotlib")
        st.info("""
        Para habilitar WordClouds, asegurate de que en requirements.txt tengas:
        wordcloud==1.9.2
        matplotlib==3.7.0
        Pillow==10.0.0
        """)
    
    # Analisis recientes
    st.markdown("---")
    st.subheader("Analisis Recientes")
    
    for idx, row in df.head(10).iterrows():
        sentiment_value = row['prediction_label']
        probability = row.get('positive_probability', 'N/A')
        
        # Formatear probabilidad
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

    # Informacion tecnica
    st.markdown("---")
    st.subheader("Informacion Tecnica")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Estructura del Dataset:")
        st.json({
            "total_registros": total_tweets,
            "positivos": positive_count,
            "negativos": negative_count,
            "porcentaje_positivo": f"{positive_percentage:.2f}%",
            "archivos_cargados": len(glob.glob("stream_output/*.parquet")) + len(glob.glob("stream_output/*.snappy.parquet"))
        })

    with col2:
        st.write("Estado de Dependencias:")
        status_data = {
            "pandas": "Disponible" if PANDAS_AVAILABLE else "No disponible",
            "plotly": "Disponible" if PLOTLY_AVAILABLE else "No disponible",
            "wordcloud": "Disponible" if WORDCLOUD_AVAILABLE else "No disponible",
            "numpy": "Disponible" if NUMPY_AVAILABLE else "No disponible"
        }
        st.json(status_data)

else:
    st.error("Pandas no esta disponible - Dashboard limitado")
    st.info("""
    El problema es que Streamlit Cloud no puede instalar pandas/pyarrow.
    
    Solucion:
    1. Usa la version compatible de pandas: pandas==1.5.3
    2. Elimina pyarrow si no es esencial
    3. Actualiza el archivo requirements.txt
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6c757d;'>"
    "Proyecto Big Data - Analisis de Sentimientos con Spark ML<br>"
    "Dashboard con WordClouds integradas"
    "</div>", 
    unsafe_allow_html=True
)
