import streamlit as st
import os
import glob
from datetime import datetime
import collections

# Configuración de la página
st.set_page_config(
    page_title="Dashboard de Analisis de Sentimientos",
    layout="wide"
)

# Título principal
st.title("Dashboard de Analisis de Sentimientos en Tiempo Real")
st.markdown("---")

# Intentar importar dependencias esenciales
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

def create_sample_data():
    """Crear datos de ejemplo robustos"""
    sample_data = {
        "tweet": [
            "Me encanta este producto es increible la calidad que ofrece",
            "No me gusta para nada muy decepcionado con el servicio",
            "Excelente atencion al cliente resolvieron mi problema rapido",
            "Pesima calidad el producto se dano en la primera semana",
            "Entrega rapida y producto en perfecto estado recomendado",
            "Mala experiencia el soporte tecnico no sabe resolver",
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

def analyze_frequent_words(text_data, title, max_words=15):
    """Analizar palabras frecuentes sin dependencias externas"""
    if not text_data:
        return None
    
    try:
        # Palabras a excluir
        stop_words = {
            'el', 'la', 'los', 'las', 'de', 'en', 'y', 'que', 'con', 'para', 
            'por', 'como', 'me', 'mi', 'un', 'una', 'unos', 'unas', 'es', 'son',
            'se', 'no', 'si', 'lo', 'le', 'al', 'del', 'su', 'sus', 'este', 
            'esta', 'estos', 'estas', 'a', 'o', 'u', 'the', 'and', 'is', 'in',
            'on', 'at', 'to', 'for', 'with', 'my', 'your', 'our', 'their'
        }
        
        # Contar palabras
        word_counts = collections.Counter()
        for text in text_data:
            if isinstance(text, str):
                words = text.lower().split()
                for word in words:
                    # Filtrar palabras
                    if (len(word) > 3 and 
                        word.isalpha() and 
                        word not in stop_words):
                        word_counts[word] += 1
        
        # Tomar las palabras mas frecuentes
        top_words = word_counts.most_common(max_words)
        
        if not top_words:
            return None
            
        # Crear datos para el grafico
        words, counts = zip(*top_words)
        
        if PLOTLY_AVAILABLE:
            # Grafico de barras horizontal con Plotly
            fig = px.bar(
                x=list(counts),
                y=list(words),
                orientation='h',
                title=title,
                labels={'x': 'Frecuencia', 'y': 'Palabras'},
                color=list(counts),
                color_continuous_scale='viridis'
            )
            fig.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
            return fig
        else:
            # Tabla basica con Streamlit
            st.write(f"**{title}**")
            for word, count in top_words:
                st.write(f"- {word}: {count}")
            return None
            
    except Exception as e:
        st.error(f"Error analizando palabras: {str(e)}")
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
    
    # Visualizaciones principales
    st.markdown("---")
    st.subheader("Visualizaciones")
    
    if PLOTLY_AVAILABLE:
        # Grafico Plotly de torta
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
        
        # Grafico de probabilidades
        if 'positive_probability' in df.columns:
            fig_hist = px.histogram(
                df, 
                x='positive_probability',
                title="Distribucion de Probabilidades",
                labels={'positive_probability': 'Probabilidad Positiva'},
                nbins=20
            )
            st.plotly_chart(fig_hist, use_container_width=True)
    else:
        # Graficos basicos
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Distribucion de Sentimientos**")
            st.write(f"Positivos: {positive_count} ({positive_percentage:.1f}%)")
            st.progress(positive_percentage/100)
        with col2:
            st.write(f"**Negativos:** {negative_count} ({100-positive_percentage:.1f}%)")
            st.progress((100-positive_percentage)/100)
    
    # ANALISIS DE PALABRAS FRECUENTES
    st.markdown("---")
    st.subheader("Analisis de Texto - Palabras Frecuentes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Palabras frecuentes positivas
        positive_tweets = df[df['prediction_label'] == 1]['tweet'].tolist()
        words_pos = analyze_frequent_words(positive_tweets, "Palabras Mas Frecuentes - Sentimientos Positivos")
        if words_pos:
            st.plotly_chart(words_pos, use_container_width=True)
        else:
            st.info("No hay datos suficientes para analizar palabras positivas")
    
    with col2:
        # Palabras frecuentes negativas
        negative_tweets = df[df['prediction_label'] == 0]['tweet'].tolist()
        words_neg = analyze_frequent_words(negative_tweets, "Palabras Mas Frecuentes - Sentimientos Negativos")
        if words_neg:
            st.plotly_chart(words_neg, use_container_width=True)
        else:
            st.info("No hay datos suficientes para analizar palabras negativas")
    
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
        st.write("**Estructura del Dataset:**")
        st.json({
            "total_registros": total_tweets,
            "positivos": positive_count,
            "negativos": negative_count,
            "porcentaje_positivo": f"{positive_percentage:.2f}%"
        })

    with col2:
        st.write("**Estado de Dependencias:**")
        status_data = {
            "pandas": "Disponible" if PANDAS_AVAILABLE else "No disponible",
            "plotly": "Disponible" if PLOTLY_AVAILABLE else "No disponible"
        }
        st.json(status_data)

else:
    st.error("Pandas no esta disponible - Dashboard limitado")
    st.info("""
    Para solucionar este problema:
    1. Asegurate de que requirements.txt solo tiene librerias compatibles
    2. Usa versiones conocidas que funcionen en Streamlit Cloud
    3. Evita librerias que requieran compilacion
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6c757d;'>"
    "Proyecto Big Data - Analisis de Sentimientos con Spark ML<br>"
    "Dashboard compatible con Streamlit Cloud"
    "</div>", 
    unsafe_allow_html=True
)
