import streamlit as st
import pandas as pd
import quantstats as qs
import numpy as np
import io
import base64
from datetime import datetime
import plotly.graph_objects as go
import yfinance as yf

# Page config
st.set_page_config(
    page_title="QuantStats Pro Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #16213e 0%, #0f1419 100%);
        border-right: 1px solid #2d3250;
    }
    
    h1, h2, h3 { color: #00d4ff !important; font-weight: 700 !important; }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.05) 0%, rgba(0, 153, 204, 0.05) 100%);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    }
    
    .insight-box {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.05) 0%, rgba(0, 212, 255, 0.05) 100%);
        border-left: 4px solid #00ff88;
        border-radius: 8px;
        padding: 15px 20px;
        margin: 15px 0;
        color: #e0e0e0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(255, 153, 0, 0.05) 0%, rgba(255, 102, 0, 0.05) 100%);
        border-left: 4px solid #ff9900;
        border-radius: 8px;
        padding: 15px 20px;
        margin: 15px 0;
        color: #e0e0e0;
    }
    
    .guide-box {
        background: linear-gradient(135deg, rgba(138, 43, 226, 0.05) 0%, rgba(75, 0, 130, 0.05) 100%);
        border-left: 4px solid #8a2be2;
        border-radius: 8px;
        padding: 15px 20px;
        margin: 15px 0;
        color: #e0e0e0;
    }
    
    [data-testid="stMetricValue"] {
        color: #00ff88 !important;
        font-size: 32px !important;
        font-weight: 700 !important;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #00d4ff 0%, #0099cc 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 35px;
        font-weight: 600;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #00ff88 0%, #00d4ff 100%);
        transform: translateY(-2px);
    }
    
    .stExpander {
        background: rgba(45, 50, 80, 0.2);
        border: 1px solid rgba(0, 212, 255, 0.1);
        border-radius: 10px;
    }
    
    [data-testid="stExpander"] summary {
        color: #00d4ff !important;
        font-weight: 600;
    }
    
    .section-header {
        background: linear-gradient(90deg, rgba(0, 212, 255, 0.1) 0%, transparent 100%);
        border-left: 4px solid #00d4ff;
        padding: 12px 20px;
        border-radius: 8px;
        margin: 20px 0 15px 0;
    }
    
    .footer {
        text-align: center;
        padding: 30px 20px;
        margin-top: 50px;
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.03) 0%, rgba(0, 153, 204, 0.03) 100%);
        border-top: 2px solid rgba(0, 212, 255, 0.2);
        border-radius: 15px;
    }
    
    .footer-link {
        color: #00d4ff;
        text-decoration: none;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .footer-link:hover {
        color: #00ff88;
        text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
    }
    
    [data-testid="stFileUploader"] {
        background: rgba(45, 50, 80, 0.3);
        border: 2px dashed #00d4ff;
        border-radius: 12px;
        padding: 25px;
    }
    
    .step-number {
        display: inline-block;
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
        color: white;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        text-align: center;
        line-height: 28px;
        font-weight: 700;
        margin-right: 10px;
        box-shadow: 0 4px 10px rgba(0, 212, 255, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'preferences' not in st.session_state:
    st.session_state.preferences = {
        'show_insights': True,
        'show_benchmark_comparison': True,
        'metrics': {
            'basic': True,
            'risk': True,
            'drawdown': True,
            'returns': True
        },
        'charts': {
            'cumulative_returns': True,
            'monthly_heatmap': True,
            'distribution': True,
            'drawdown': True,
            'rolling_stats': True,
            'yearly_returns': True,
            'qq_plot': False,
            'log_returns': False,
            'rolling_vol': True,
            'rolling_sharpe': True,
            'rolling_beta': True
        },
        'advanced': {
            'monte_carlo': False,
            'time_analysis': True,
            'statistical_edge': True
        }
    }

# Helper function
def get_insight(metric_name, value):
    insights = {
        'sharpe': {
            'good': (2.0, "🌟 ¡Excelente rentabilidad ajustada al riesgo! Fuerte rendimiento relativo a la volatilidad."),
            'ok': (1.0, "✅ Buena performance ajustada al riesgo. Espacio para optimización."),
            'bad': (0, "⚠️ Baja rentabilidad ajustada al riesgo. Revisa la gestión de riesgo.")
        },
        'max_dd': {
            'good': (-10, "🌟 ¡Excelente control de caídas! Drawdown muy manejable."),
            'ok': (-20, "✅ Drawdown moderado. Aceptable para la mayoría de estrategias."),
            'bad': (-100, "⚠️ Drawdown significativo. Podría probar la paciencia del inversor.")
        },
        'cagr': {
            'good': (15, "🌟 ¡Crecimiento compuesto excepcional! Rendimiento consistentemente fuerte."),
            'ok': (8, "✅ Tasa de crecimiento sólida a largo plazo. Superando la inflación."),
            'bad': (0, "⚠️ Crecimiento bajo rendimiento. Puede no cumplir objetivos.")
        }
    }
    
    if metric_name in insights:
        thresholds = insights[metric_name]
        if value >= thresholds['good'][0]:
            return f"<div class='insight-box'><b>💡 Insight:</b> {thresholds['good'][1]}</div>"
        elif value >= thresholds['ok'][0]:
            return f"<div class='insight-box'><b>💡 Insight:</b> {thresholds['ok'][1]}</div>"
        else:
            return f"<div class='warning-box'><b>⚠️ Insight:</b> {thresholds['bad'][1]}</div>"
    return ""

# Title
st.markdown("<h1 style='text-align: center; margin-bottom: 0; font-size: 48px;'>📊 QuantStats Pro Analytics</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #00d4ff; font-size: 18px; margin-top: 5px;'>Análisis Cuantitativo Profesional de Estrategias</p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("### 📁 Importar Datos")
    uploaded_file = st.file_uploader(
        "Sube los Retornos de tu Estrategia",
        type=['csv', 'xlsx', 'txt'],
        help="Se requieren columnas de fecha y retornos"
    )
    
    if uploaded_file:
        st.markdown("---")
        st.markdown("### 🎯 Configuración de Benchmark")
        
        benchmark_type = st.radio(
            "Tipo de Benchmark",
            ["Predefinido (yfinance)", "CSV Personalizado", "Ninguno"],
            index=0
        )
        
        benchmark_option = None
        custom_benchmark = None
        benchmark_file = None
        
        if benchmark_type == "Predefinido (yfinance)":
            benchmark_option = st.selectbox(
                "Selecciona Benchmark",
                ["SPY", "QQQ", "IWM", "EFA", "AGG", "GLD", "^GSPC", "^IXIC", "BTC-USD", "ETH-USD"],
                index=0,
                help="Datos descargados automáticamente de Yahoo Finance"
            )
            
            col_a, col_b = st.columns(2)
            with col_a:
                start_date = st.date_input("Fecha Inicio", value=None, help="Dejar vacío para usar rango completo")
            with col_b:
                end_date = st.date_input("Fecha Fin", value=None, help="Dejar vacío para usar rango completo")
            
            st.info(f"📊 Se descargará: **{benchmark_option}** desde Yahoo Finance")
        
        elif benchmark_type == "CSV Personalizado":
            benchmark_file = st.file_uploader(
                "Sube Benchmark CSV",
                type=['csv', 'xlsx'],
                help="Archivo con columnas de fecha y retornos del benchmark"
            )
        
        else:
            st.info("ℹ️ Análisis sin benchmark. Solo métricas de la estrategia.")
        
        st.markdown("---")
        
        rf_rate = st.number_input(
            "Tasa Libre de Riesgo (%)",
            min_value=0.0,
            max_value=10.0,
            value=4.5,
            step=0.1
        ) / 100
        
        periods_per_year = st.selectbox("Períodos/Año", [252, 365, 12, 52, 1], index=0)
        
        st.markdown("---")
        st.markdown("### 🎨 Personalizar Visualización")
        
        with st.expander("📊 Métricas a Mostrar", expanded=False):
            st.session_state.preferences['metrics']['basic'] = st.checkbox(
                "Métricas Básicas (Retorno, CAGR, Sharpe)", 
                value=st.session_state.preferences['metrics']['basic']
            )
            st.session_state.preferences['metrics']['risk'] = st.checkbox(
                "Métricas de Riesgo (VaR, CVaR, Volatilidad)", 
                value=st.session_state.preferences['metrics']['risk']
            )
            st.session_state.preferences['metrics']['drawdown'] = st.checkbox(
                "Métricas de Drawdown", 
                value=st.session_state.preferences['metrics']['drawdown']
            )
            st.session_state.preferences['metrics']['returns'] = st.checkbox(
                "Análisis de Retornos (Win Rate, Mejor/Peor)", 
                value=st.session_state.preferences['metrics']['returns']
            )
        
        with st.expander("📈 Gráficos a Mostrar", expanded=False):
            st.session_state.preferences['charts']['cumulative_returns'] = st.checkbox(
                "Retornos Acumulados", 
                value=st.session_state.preferences['charts']['cumulative_returns']
            )
            st.session_state.preferences['charts']['monthly_heatmap'] = st.checkbox(
                "Mapa de Calor Mensual", 
                value=st.session_state.preferences['charts']['monthly_heatmap']
            )
            st.session_state.preferences['charts']['distribution'] = st.checkbox(
                "Distribución de Retornos", 
                value=st.session_state.preferences['charts']['distribution']
            )
            st.session_state.preferences['charts']['drawdown'] = st.checkbox(
                "Gráfico de Drawdown", 
                value=st.session_state.preferences['charts']['drawdown']
            )
            st.session_state.preferences['charts']['rolling_stats'] = st.checkbox(
                "Estadísticas Móviles", 
                value=st.session_state.preferences['charts']['rolling_stats']
            )
            st.session_state.preferences['charts']['yearly_returns'] = st.checkbox(
                "Retornos Anuales", 
                value=st.session_state.preferences['charts']['yearly_returns']
            )
            st.session_state.preferences['charts']['qq_plot'] = st.checkbox(
                "Gráfico QQ (Normalidad)", 
                value=st.session_state.preferences['charts']['qq_plot']
            )
            st.session_state.preferences['charts']['log_returns'] = st.checkbox(
                "Retornos Logarítmicos", 
                value=st.session_state.preferences['charts']['log_returns']
            )
            st.session_state.preferences['charts']['rolling_vol'] = st.checkbox(
                "Volatilidad Móvil", 
                value=st.session_state.preferences['charts']['rolling_vol']
            )
            st.session_state.preferences['charts']['rolling_sharpe'] = st.checkbox(
                "Sharpe Móvil", 
                value=st.session_state.preferences['charts']['rolling_sharpe']
            )
            st.session_state.preferences['charts']['rolling_beta'] = st.checkbox(
                "Beta Móvil (si hay benchmark)", 
                value=st.session_state.preferences['charts']['rolling_beta']
            )
        
        with st.expander("⚙️ Funciones Avanzadas", expanded=False):
            st.session_state.preferences['show_insights'] = st.checkbox(
                "Mostrar Insights IA", 
                value=st.session_state.preferences['show_insights']
            )
            st.session_state.preferences['show_benchmark_comparison'] = st.checkbox(
                "Comparación con Benchmark", 
                value=st.session_state.preferences['show_benchmark_comparison']
            )
            st.session_state.preferences['advanced']['statistical_edge'] = st.checkbox(
                "Análisis de Ventaja Estadística", 
                value=st.session_state.preferences['advanced']['statistical_edge']
            )
            st.session_state.preferences['advanced']['time_analysis'] = st.checkbox(
                "Análisis Temporal", 
                value=st.session_state.preferences['advanced']['time_analysis']
            )
            st.session_state.preferences['advanced']['monte_carlo'] = st.checkbox(
                "Simulación Monte Carlo", 
                value=st.session_state.preferences['advanced']['monte_carlo']
            )
        
        st.markdown("---")
        
        st.markdown("### 🎯 Presets Rápidos")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Esencial", use_container_width=True):
                st.session_state.preferences['charts'] = {k: k in ['cumulative_returns', 'monthly_heatmap', 'drawdown', 'distribution'] for k in st.session_state.preferences['charts']}
                st.rerun()
        with col2:
            if st.button("🔬 Completo", use_container_width=True):
                st.session_state.preferences['charts'] = {k: True for k in st.session_state.preferences['charts']}
                st.session_state.preferences['advanced'] = {k: True for k in st.session_state.preferences['advanced']}
                st.rerun()
    
    st.markdown("---")
    st.markdown("<p style='color: #666; font-size: 11px; text-align: center;'>v2.0 Professional Edition</p>", unsafe_allow_html=True)

# Main content
if uploaded_file is None:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            <div style='text-align: center; padding: 60px 30px; background: linear-gradient(135deg, rgba(0, 212, 255, 0.05) 0%, rgba(0, 153, 204, 0.05) 100%); border-radius: 20px; margin-top: 30px; border: 1px solid rgba(0, 212, 255, 0.2);'>
                <h2 style='color: #00d4ff; margin-bottom: 25px; font-size: 36px;'>Bienvenido a Pro Analytics</h2>
                <p style='color: #a0a0c0; font-size: 18px; line-height: 1.8; margin-bottom: 30px;'>
                    Plataforma profesional de análisis cuantitativo completamente personalizable
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📚 Guía de Uso Rápida")
        
        st.markdown("""
            <div class='guide-box'>
                <p style='margin: 0 0 15px 0;'><span class='step-number'>1</span><b>Prepara tus datos:</b> Archivo CSV/Excel con dos columnas: fecha y retornos (en decimal 0.01 o porcentaje 1.0)</p>
                <p style='margin: 0 0 15px 0;'><span class='step-number'>2</span><b>Sube el archivo:</b> Usa el panel lateral izquierdo para cargar tu CSV de estrategia</p>
                <p style='margin: 0 0 15px 0;'><span class='step-number'>3</span><b>Configura benchmark:</b> Elige descargar desde Yahoo Finance o sube tu propio benchmark</p>
                <p style='margin: 0 0 15px 0;'><span class='step-number'>4</span><b>Personaliza vista:</b> Selecciona qué métricas y gráficos mostrar</p>
                <p style='margin: 0;'><span class='step-number'>5</span><b>Analiza y exporta:</b> Explora resultados y descarga reportes profesionales</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### ✨ Características Principales")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("""
                <div class='metric-card'>
                    <h4 style='color: #00d4ff; margin-top: 0;'>📊 Métricas Completas</h4>
                    <ul style='color: #a0a0c0; line-height: 1.8;'>
                        <li>Sharpe, Sortino, Calmar</li>
                        <li>VaR, CVaR, Kelly Criterion</li>
                        <li>Drawdown y recuperación</li>
                        <li>Win rate y payoff ratio</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div class='metric-card'>
                    <h4 style='color: #00d4ff; margin-top: 0;'>🎨 Personalización Total</h4>
                    <ul style='color: #a0a0c0; line-height: 1.8;'>
                        <li>Elige qué ver exactamente</li>
                        <li>Presets rápidos (Esencial/Completo)</li>
                        <li>Insights automáticos con IA</li>
                        <li>Interfaz adaptable</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        with col_b:
            st.markdown("""
                <div class='metric-card'>
                    <h4 style='color: #00d4ff; margin-top: 0;'>📈 Visualizaciones Pro</h4>
                    <ul style='color: #a0a0c0; line-height: 1.8;'>
                        <li>Mapas de calor mensuales</li>
                        <li>Análisis de distribución</li>
                        <li>Gráficos de drawdown</li>
                        <li>Estadísticas móviles</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div class='metric-card'>
                    <h4 style='color: #00d4ff; margin-top: 0;'>🎯 Benchmarks Flexibles</h4>
                    <ul style='color: #a0a0c0; line-height: 1.8;'>
                        <li>Yahoo Finance (SPY, QQQ, etc.)</li>
                        <li>Subir benchmark personalizado</li>
                        <li>Control de rango de fechas</li>
                        <li>Comparación detallada</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("""
            <div style='text-align: center; padding: 30px; background: rgba(138, 43, 226, 0.05); border-radius: 15px; border: 1px solid rgba(138, 43, 226, 0.2);'>
                <p style='color: #00d4ff; font-size: 18px; font-weight: 600; margin: 0 0 10px 0;'>
                    👈 Comienza subiendo tu archivo CSV en el panel lateral
                </p>
                <p style='color: #a0a0c0; font-size: 13px; margin: 5px 0 0 0;'>
                    📦 Requiere: <code>streamlit quantstats yfinance pandas numpy plotly</code>
                </p>
            </div>
        """, unsafe_allow_html=True)
else:
    try:
        # Read file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file, sep='\t')
        
        st.success(f"✅ Cargado {uploaded_file.name} • {len(df)} observaciones")
        
        # Column selection
        col1, col2 = st.columns(2)
        with col1:
            date_col = st.selectbox("Columna de Fecha", df.columns, index=0)
        with col2:
            returns_col = st.selectbox("Columna de Retornos", df.columns, index=1 if len(df.columns) > 1 else 0)
        
        # Process data
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
        returns = df[returns_col].dropna()
        
        if returns.abs().mean() > 1:
            returns = returns / 100
        
        qs.extend_pandas()
        
        # Fetch benchmark
        benchmark = None
        bench_name = "Benchmark"
        
        if benchmark_type == "Predefinido (yfinance)" and benchmark_option:
            bench_name = benchmark_option
            
            with st.spinner(f"📥 Descargando {bench_name} desde Yahoo Finance..."):
                try:
                    if start_date and end_date:
                        bench_data = yf.download(benchmark_option, start=start_date, end=end_date, progress=False, auto_adjust=True)
                    else:
                        bench_data = yf.download(benchmark_option, start=returns.index.min(), end=returns.index.max(), progress=False, auto_adjust=True)
                    
                    if not bench_data.empty and len(bench_data) > 0:
                        if 'Close' in bench_data.columns:
                            benchmark = bench_data['Close'].pct_change().dropna()
                        else:
                            benchmark = bench_data.iloc[:, 0].pct_change().dropna()
                        
                        if isinstance(benchmark, pd.DataFrame):
                            benchmark = benchmark.iloc[:, 0]
                        
                        col_info1, col_info2, col_info3 = st.columns(3)
                        with col_info1:
                            st.success(f"✅ {bench_name} descargado")
                        with col_info2:
                            st.info(f"📅 {len(benchmark)} observaciones")
                        with col_info3:
                            bench_return = (1 + benchmark).prod() - 1
                            st.info(f"📈 Retorno: {bench_return*100:.2f}%")
                        
                        with st.expander("📊 Vista Previa del Benchmark", expanded=False):
                            preview_df = pd.DataFrame({
                                'Fecha': benchmark.index[-10:],
                                'Retorno': [f"{r*100:.4f}%" for r in benchmark.values[-10:]]
                            })
                            st.dataframe(preview_df, use_container_width=True, hide_index=True)
                    else:
                        st.error(f"❌ No se pudieron obtener datos de {bench_name}")
                        benchmark = None
                except Exception as e:
                    st.error(f"❌ Error al descargar {bench_name}")
                    with st.expander("🔍 Detalles del error"):
                        st.code(str(e))
                        st.info("💡 Consejos:")
                        st.markdown("""
                        - Verifica que el símbolo sea correcto (ej: SPY, QQQ, ^GSPC)
                        - Algunos símbolos requieren formato especial (ej: BTC-USD para Bitcoin)
                        - Yahoo Finance puede tener datos limitados para ciertos activos
                        - Intenta con un rango de fechas diferente
                        """)
                    benchmark = None
        
        elif benchmark_type == "CSV Personalizado" and benchmark_file:
            try:
                if benchmark_file.name.endswith('.csv'):
                    bench_df = pd.read_csv(benchmark_file)
                else:
                    bench_df = pd.read_excel(benchmark_file)
                
                st.success(f"✅ Benchmark cargado: {benchmark_file.name}")
                
                with st.expander("📄 Vista Previa del Benchmark"):
                    st.dataframe(bench_df.head(), use_container_width=True)
                
                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    bench_date_col = st.selectbox("Columna Fecha (Benchmark)", bench_df.columns, index=0, key="bench_date")
                with col_b2:
                    bench_ret_col = st.selectbox("Columna Retornos (Benchmark)", bench_df.columns, index=1 if len(bench_df.columns) > 1 else 0, key="bench_ret")
                
                bench_df[bench_date_col] = pd.to_datetime(bench_df[bench_date_col])
                bench_df.set_index(bench_date_col, inplace=True)
                benchmark = bench_df[bench_ret_col].dropna()
                
                if benchmark.abs().mean() > 1:
                    benchmark = benchmark / 100
                
                bench_name = benchmark_file.name.split('.')[0]
                st.info(f"📊 Benchmark procesado: {len(benchmark)} observaciones")
                
            except Exception as e:
                st.error(f"❌ Error al procesar benchmark: {str(e)}")
                benchmark = None
        
        st.markdown("---")
        
        # Calculate metrics
        prefs = st.session_state.preferences
        
        # === BASIC METRICS ===
        if prefs['metrics']['basic']:
            st.markdown("<div class='section-header'><h3 style='margin:0;'>📊 Métricas de Rendimiento</h3></div>", unsafe_allow_html=True)
            
            total_return = qs.stats.comp(returns)
            cagr = qs.stats.cagr(returns, rf=rf_rate, periods=periods_per_year)
            sharpe = qs.stats.sharpe(returns, rf=rf_rate, periods=periods_per_year)
            sortino = qs.stats.sortino(returns, rf=rf_rate, periods=periods_per_year)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Retorno Total", f"{total_return*100:.2f}%")
            with col2:
                st.metric("CAGR", f"{cagr*100:.2f}%")
            with col3:
                st.metric("Ratio Sharpe", f"{sharpe:.2f}")
            with col4:
                st.metric("Ratio Sortino", f"{sortino:.2f}")
            
            if prefs['show_insights']:
                st.markdown(get_insight('sharpe', sharpe), unsafe_allow_html=True)
                st.markdown(get_insight('cagr', cagr*100), unsafe_allow_html=True)
        
        # === RISK METRICS ===
        if prefs['metrics']['risk']:
            st.markdown("<div class='section-header'><h3 style='margin:0;'>⚠️ Métricas de Riesgo</h3></div>", unsafe_allow_html=True)
            
            volatility = qs.stats.volatility(returns, periods=periods_per_year)
            var_95 = qs.stats.var(returns)
            cvar_95 = qs.stats.cvar(returns)
            kelly = qs.stats.kelly_criterion(returns)
            skew = qs.stats.skew(returns)
            kurtosis = qs.stats.kurtosis(returns)
            
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1:
                st.metric("Volatilidad", f"{volatility*100:.2f}%")
            with col2:
                st.metric("VaR (95%)", f"{var_95*100:.2f}%")
            with col3:
                st.metric("CVaR (95%)", f"{cvar_95*100:.2f}%")
            with col4:
                st.metric("Kelly %", f"{kelly*100:.1f}%")
            with col5:
                st.metric("Asimetría", f"{skew:.2f}")
            with col6:
                st.metric("Curtosis", f"{kurtosis:.2f}")
            
            if prefs['show_insights'] and kelly > 0.25:
                st.markdown(f"<div class='warning-box'><b>⚠️ Tamaño de Posición:</b> Kelly sugiere {kelly*100:.1f}% de asignación. Considera 0.5x Kelly ({kelly*50:.1f}%) para implementación práctica.</div>", unsafe_allow_html=True)
        
        # === DRAWDOWN METRICS ===
        if prefs['metrics']['drawdown']:
            st.markdown("<div class='section-header'><h3 style='margin:0;'>📉 Métricas de Drawdown</h3></div>", unsafe_allow_html=True)
            
            max_dd = qs.stats.max_drawdown(returns)
            calmar = qs.stats.calmar(returns, periods=periods_per_year)
            
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            
            is_drawdown = drawdown < 0
            dd_periods = []
            if is_drawdown.any():
                dd_start = None
                for i, (date, is_dd) in enumerate(is_drawdown.items()):
                    if is_dd and dd_start is None:
                        dd_start = i
                    elif not is_dd and dd_start is not None:
                        dd_periods.append(drawdown.iloc[dd_start:i])
                        dd_start = None
                if dd_start is not None:
                    dd_periods.append(drawdown.iloc[dd_start:])
            
            avg_dd = np.mean([dd.min() for dd in dd_periods]) if dd_periods else 0
            avg_dd_days = np.mean([len(dd) for dd in dd_periods]) if dd_periods else 0
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Drawdown Máximo", f"{max_dd*100:.2f}%")
            with col2:
                st.metric("Drawdown Promedio", f"{avg_dd*100:.2f}%")
            with col3:
                st.metric("Días Promedio DD", f"{avg_dd_days:.0f}")
            with col4:
                st.metric("Ratio Calmar", f"{calmar:.2f}")
            
            if prefs['show_insights']:
                st.markdown(get_insight('max_dd', max_dd*100), unsafe_allow_html=True)
        
        # === RETURNS ANALYSIS ===
        if prefs['metrics']['returns']:
            st.markdown("<div class='section-header'><h3 style='margin:0;'>💰 Análisis de Retornos</h3></div>", unsafe_allow_html=True)
            
            win_rate = qs.stats.win_rate(returns)
            best = qs.stats.best(returns)
            worst = qs.stats.worst(returns)
            payoff = qs.stats.payoff_ratio(returns)
            profit_factor = qs.stats.profit_factor(returns)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Tasa de Acierto", f"{win_rate*100:.1f}%")
            with col2:
                st.metric("Mejor Día", f"{best*100:.2f}%")
            with col3:
                st.metric("Peor Día", f"{worst*100:.2f}%")
            with col4:
                st.metric("Ratio Payoff", f"{payoff:.2f}")
            with col5:
                st.metric("Factor de Beneficio", f"{profit_factor:.2f}")
        
        # === BENCHMARK COMPARISON ===
        if benchmark is not None and prefs['show_benchmark_comparison']:
            st.markdown("<div class='section-header'><h3 style='margin:0;'>🎯 vs Benchmark</h3></div>", unsafe_allow_html=True)
            
            bench_return = qs.stats.comp(benchmark)
            bench_sharpe = qs.stats.sharpe(benchmark, rf=rf_rate, periods=periods_per_year)
            beta = qs.stats.beta(returns, benchmark)
            alpha = qs.stats.alpha(returns, benchmark, rf=rf_rate, periods=periods_per_year)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(f"Retorno {bench_name}", f"{bench_return*100:.2f}%",
                         delta=f"{(total_return - bench_return)*100:.2f}% Estrategia")
            with col2:
                st.metric(f"Sharpe {bench_name}", f"{bench_sharpe:.2f}",
                         delta=f"{(sharpe - bench_sharpe):.2f} Estrategia")
            with col3:
                st.metric("Beta", f"{beta:.2f}")
            with col4:
                st.metric("Alpha (Anual)", f"{alpha*100:.2f}%")
        
        st.markdown("---")
        
        # === CHARTS SECTION ===
        st.markdown("## 📈 Análisis Visual")
        
        if prefs['charts']['cumulative_returns']:
            with st.expander("📈 Retornos Acumulados", expanded=True):
                fig = qs.plots.returns(returns, benchmark=benchmark, show=False, figsize=(14, 6))
                st.pyplot(fig)
        
        charts_to_show = []
        
        if prefs['charts']['monthly_heatmap']:
            charts_to_show.append(('monthly_heatmap', "🗓️ Mapa de Calor Mensual"))
        if prefs['charts']['distribution']:
            charts_to_show.append(('distribution', "📊 Distribución de Retornos"))
        if prefs['charts']['drawdown']:
            charts_to_show.append(('drawdown', "📉 Períodos de Drawdown"))
        if prefs['charts']['yearly_returns']:
            charts_to_show.append(('yearly_returns', "📅 Retornos Anuales"))
        if prefs['charts']['qq_plot']:
            charts_to_show.append(('qq_plot', "📊 Gráfico QQ (Normalidad)"))
        if prefs['charts']['log_returns']:
            charts_to_show.append(('log_returns', "📈 Retornos Logarítmicos"))
        if prefs['charts']['rolling_vol']:
            charts_to_show.append(('rolling_vol', "📊 Volatilidad Móvil"))
        if prefs['charts']['rolling_sharpe']:
            charts_to_show.append(('rolling_sharpe', "📈 Sharpe Móvil"))
        if prefs['charts']['rolling_beta'] and benchmark is not None:
            charts_to_show.append(('rolling_beta', "🎯 Beta Móvil"))
        if prefs['charts']['rolling_stats']:
            charts_to_show.append(('rolling_stats', "📊 Estadísticas Móviles"))
        
        for i in range(0, len(charts_to_show), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(charts_to_show):
                    chart_type, chart_title = charts_to_show[i + j]
                    with cols[j]:
                        with st.expander(chart_title, expanded=False):
                            if chart_type == 'monthly_heatmap':
                                fig = qs.plots.monthly_heatmap(returns, show=False, figsize=(10, 7))
                            elif chart_type == 'distribution':
                                fig = qs.plots.histogram(returns, show=False, figsize=(10, 6))
                            elif chart_type == 'drawdown':
                                fig = qs.plots.drawdowns_periods(returns, show=False, figsize=(10, 6))
                            elif chart_type == 'yearly_returns':
                                fig = qs.plots.yearly_returns(returns, benchmark=benchmark, show=False, figsize=(10, 6))
                            elif chart_type == 'qq_plot':
                                fig = qs.plots.qq(returns, show=False, figsize=(10, 6))
                            elif chart_type == 'log_returns':
                                fig = qs.plots.log_returns(returns, show=False, figsize=(10, 6))
                            elif chart_type == 'rolling_vol':
                                fig = qs.plots.rolling_volatility(returns, period=periods_per_year, show=False, figsize=(10, 6))
                            elif chart_type == 'rolling_sharpe':
                                fig = qs.plots.rolling_sharpe(returns, rf=rf_rate, period=periods_per_year, show=False, figsize=(10, 6))
                            elif chart_type == 'rolling_beta':
                                fig = qs.plots.rolling_beta(returns, benchmark, show=False, figsize=(10, 6))
                            elif chart_type == 'rolling_stats':
                                fig = qs.plots.rolling_stats(returns, show=False, figsize=(10, 6))
                            st.pyplot(fig)
        
        # === ADVANCED FEATURES ===
        if any(prefs['advanced'].values()):
            st.markdown("---")
            st.markdown("## 🔬 Análisis Avanzado")
        
        if prefs['advanced']['statistical_edge']:
            with st.expander("📊 Análisis de Ventaja Estadística", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                avg_win = qs.stats.avg_win(returns)
                avg_loss = qs.stats.avg_loss(returns)
                
                returns_sign = np.sign(returns)
                consecutive_wins = 0
                consecutive_losses = 0
                current_win_streak = 0
                current_loss_streak = 0
                
                for r in returns_sign:
                    if r > 0:
                        current_win_streak += 1
                        current_loss_streak = 0
                        consecutive_wins = max(consecutive_wins, current_win_streak)
                    elif r < 0:
                        current_loss_streak += 1
                        current_win_streak = 0
                        consecutive_losses = max(consecutive_losses, current_loss_streak)
                    else:
                        current_win_streak = 0
                        current_loss_streak = 0
                
                with col1:
                    st.metric("Ganancia Promedio", f"{avg_win*100:.2f}%")
                    st.metric("Rachas Ganadoras", f"{consecutive_wins:.0f}")
                with col2:
                    st.metric("Pérdida Promedio", f"{avg_loss*100:.2f}%")
                    st.metric("Rachas Perdedoras", f"{consecutive_losses:.0f}")
                with col3:
                    st.metric("Ratio Gan/Pérd", f"{(avg_win/abs(avg_loss)):.2f}")
                    st.metric("Tasa de Acierto", f"{win_rate*100:.1f}%")
                
                if prefs['show_insights']:
                    if win_rate > 0.5 and payoff > 1.5:
                        st.markdown("<div class='insight-box'><b>🌟 Ventaja Fuerte:</b> Alta tasa de acierto + payoff favorable = ventaja estratégica robusta.</div>", unsafe_allow_html=True)
                    elif payoff > 2:
                        st.markdown("<div class='insight-box'><b>💡 Ventaja Asimétrica:</b> Ratio payoff fuerte sugiere características de seguimiento de tendencias.</div>", unsafe_allow_html=True)
        
        if prefs['advanced']['time_analysis']:
            with st.expander("🔄 Análisis Temporal", expanded=False):
                monthly_rets = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
                positive_months = (monthly_rets > 0).sum()
                total_months = len(monthly_rets)
                monthly_win_rate = (positive_months / total_months) * 100
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Meses Positivos", f"{positive_months}/{total_months}")
                with col2:
                    st.metric("Tasa Mensual", f"{monthly_win_rate:.1f}%")
                with col3:
                    st.metric("Mejor Mes", f"{monthly_rets.max()*100:.2f}%")
                with col4:
                    st.metric("Peor Mes", f"{monthly_rets.min()*100:.2f}%")
                
                st.markdown("#### 📊 Distribución de Retornos Mensuales")
                fig = qs.plots.monthly_returns(returns, show=False, figsize=(14, 6))
                st.pyplot(fig)
        
        if prefs['advanced']['monte_carlo']:
            with st.expander("🎲 Simulación Monte Carlo", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    n_sims = st.slider("Simulaciones", 100, 3000, 1000, 100)
                with col2:
                    n_days = st.slider("Días", 30, 500, 252, 30)
                with col3:
                    run_sim = st.button("🚀 Ejecutar Simulación", use_container_width=True)
                
                if run_sim:
                    with st.spinner("Ejecutando Monte Carlo..."):
                        mu = returns.mean()
                        sigma = returns.std()
                        
                        simulations = np.zeros((n_days, n_sims))
                        for i in range(n_sims):
                            daily_returns = np.random.normal(mu, sigma, n_days)
                            simulations[:, i] = (1 + daily_returns).cumprod()
                        
                        fig = go.Figure()
                        
                        for i in range(min(50, n_sims)):
                            fig.add_trace(go.Scatter(
                                x=list(range(n_days)), y=simulations[:, i],
                                mode='lines', line=dict(color='rgba(0, 212, 255, 0.1)', width=1),
                                showlegend=False, hoverinfo='skip'
                            ))
                        
                        p50 = np.percentile(simulations, 50, axis=1)
                        fig.add_trace(go.Scatter(
                            x=list(range(n_days)), y=p50,
                            mode='lines', name='Mediana',
                            line=dict(color='#00ff88', width=3)
                        ))
                        
                        fig.update_layout(
                            template='plotly_dark', height=500,
                            title=f'Monte Carlo: {n_sims} trayectorias, {n_days} días',
                            xaxis_title='Días', yaxis_title='Valor del Portafolio'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        final_values = simulations[-1, :]
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Valor Final Mediano", f"{np.median(final_values):.2f}x")
                        with col2:
                            st.metric("Prob. Beneficio", f"{(final_values > 1).mean()*100:.1f}%")
                        with col3:
                            st.metric("Percentil 95", f"{np.percentile(final_values, 95):.2f}x")
                        with col4:
                            st.metric("Percentil 5", f"{np.percentile(final_values, 5):.2f}x")
        
        # === REPORTS ===
        st.markdown("---")
        st.markdown("## 📑 Exportar Reportes")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Tabla de Métricas", use_container_width=True):
                with st.spinner("Generando..."):
                    if benchmark is not None:
                        metrics_df = qs.reports.metrics(returns, benchmark=benchmark, rf=rf_rate, display=False, mode='full')
                    else:
                        metrics_df = qs.reports.metrics(returns, mode='full', rf=rf_rate, display=False)
                    
                    st.dataframe(metrics_df, use_container_width=True)
                    csv = metrics_df.to_csv()
                    st.download_button("📥 Descargar CSV", csv, 
                                     f"metricas_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
        
        with col2:
            if st.button("📄 Reporte HTML", use_container_width=True):
                with st.spinner("Generando reporte completo..."):
                    output = io.BytesIO()
                    if benchmark is not None:
                        qs.reports.html(returns, benchmark=benchmark, rf=rf_rate, output=output)
                    else:
                        qs.reports.html(returns, output=output, rf=rf_rate)
                    
                    output.seek(0)
                    b64 = base64.b64encode(output.read()).decode()
                    href = f'<a href="data:text/html;base64,{b64}" download="reporte_{datetime.now().strftime("%Y%m%d")}.html"><button style="padding: 12px 30px; background: linear-gradient(90deg, #00ff88 0%, #00d4ff 100%); color: white; border: none; border-radius: 10px; font-weight: 600;">📥 Descargar Reporte</button></a>'
                    st.markdown(href, unsafe_allow_html=True)
        
        with col3:
            if st.button("📸 Tearsheet", use_container_width=True):
                with st.spinner("Creando tearsheet..."):
                    fig = qs.plots.snapshot(returns, show=False, figsize=(14, 10))
                    st.pyplot(fig)
    
    except Exception as e:
        st.error(f"❌ Error al procesar los datos")
        with st.expander("🔍 Detalles del error", expanded=True):
            st.code(str(e))
            st.markdown("---")
            st.markdown("### 💡 Posibles soluciones:")
            st.markdown("""
            1. **Formato del archivo**: Verifica que tu CSV tenga columnas de fecha y retornos
            2. **Formato de retornos**: Deben estar en decimal (0.01) o porcentaje (1.0)
            3. **Fechas**: Asegúrate que las fechas estén en formato reconocible (YYYY-MM-DD, DD/MM/YYYY, etc.)
            4. **Datos faltantes**: Revisa que no haya filas vacías o valores nulos excesivos
            5. **Encoding**: Si el archivo tiene caracteres especiales, guárdalo como UTF-8
            """)
            
            if 'df' in locals() and df is not None:
                st.markdown("### 📄 Vista previa de tus datos:")
                st.dataframe(df.head(10), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div class='footer'>
        <p style='color: #00d4ff; font-size: 18px; font-weight: 700; margin-bottom: 10px;'>
            📊 QuantStats Pro Analytics
        </p>
        <p style='color: #a0a0c0; font-size: 14px; margin-bottom: 15px;'>
            Análisis Cuantitativo de Nivel Institucional
        </p>
        <p style='color: #00ff88; font-size: 16px; font-weight: 600; margin: 0;'>
            Hecho por <a href='https://twitter.com/Gsnchez' class='footer-link' target='_blank'>@Gsnchez</a> | 
            <a href='https://bquantfinance.com' class='footer-link' target='_blank'>bquantfinance.com</a>
        </p>
    </div>
""", unsafe_allow_html=True)
