import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import random

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="E-comChurnnalisys",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS STYLING ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6, p, div {
        font-family: 'Inter', sans-serif;
        color: #E0E0E0;
    }
    
    /* Neon Text Gradient for Main Title */
    .title-text {
        background: linear-gradient(90deg, #00FFFF, #8A2BE2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        text-align: center;
        margin-bottom: 2rem;
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { text-shadow: 0 0 10px rgba(0, 255, 255, 0.5); }
        to { text-shadow: 0 0 20px rgba(138, 43, 226, 0.5); }
    }
    
    /* Glassmorphism Containers */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 255, 255, 0.1);
        border-color: rgba(0, 255, 255, 0.3);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(45deg, #00FFFF, #8A2BE2);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 15px rgba(138, 43, 226, 0.6);
    }
    
    /* Input Fields */
    .stNumberInput input {
        background-color: rgba(255, 255, 255, 0.05);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 5px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 5px;
        color: white;
        border: 1px solid transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #00FFFF;
        border-color: #00FFFF;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, rgba(0, 255, 255, 0.2), rgba(138, 43, 226, 0.2));
        color: cyan;
        border-color: cyan;
    }
    
</style>
""", unsafe_allow_html=True)

# --- MODEL LOADING LOGIC ---
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model, True
    except FileNotFoundError:
        # Dummy Model fallback class
        class DummyModel:
            def predict(self, X):
                # Simple logic for demo: High Recency + Low Freq = Churn
                return [1 if (x[0] > 60 and x[1] < 5) else 0 for x in X.values]
            
            def predict_proba(self, X):
                # Generate a probability geared towards the prediction
                probs = []
                for x in X.values:
                    risk = random.random()
                    if x[0] > 60: risk += 0.3
                    if x[1] < 5: risk += 0.2
                    if x[4] < 3: risk += 0.2 # Low rating
                    
                    risk = min(max(risk, 0.1), 0.95) # Clamp
                    probs.append([1-risk, risk])
                return np.array(probs)
                
        return DummyModel(), False

model, model_loaded = load_model()

# --- SIDEBAR INPUTS (Global for Tab 2, but placed here for structure) ---
with st.sidebar:
    st.markdown("### ⚙️ Configuración del Cliente")
    st.markdown("---")
    recency = st.number_input("📅 Recency Days (Días sin compra)", min_value=0, max_value=3650, value=30)
    frequency = st.number_input("📦 Purchase Frequency (Total Órdenes)", min_value=1, max_value=1000, value=5)
    monetary = st.number_input("💰 Total Spent ($ USD)", min_value=0.0, value=500.0)
    engagement = st.slider("🖱️ Avg Engagement Score", 0.0, 100.0, 50.0)
    rating = st.slider("⭐ Customer Rating", 1, 5, 4)
    age = st.slider("👤 Age", 18, 100, 35)
    
    st.markdown("---")
    st.caption(f"Status del Modelo: {'🟢 Cargado' if model_loaded else '🟡 Demo (Dummy)'}")
    st.caption("v1.0.0 - E-comChurnnalisys")

# --- MAIN CONTENT ---
st.markdown('<h1 class="title-text">E-comChurnnalisys</h1>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📜 Origen del Modelo", "🔮 Predicción en Tiempo Real"])

# --- TAB 1: ORIGEN DEL MODELO ---
with tab1:
    st.markdown("""
        <div class='glass-card'>
            <h3>🧠 E-Commerce Customer Behavior Dataset v2</h3>
            <p>El modelo predictivo nace de un riguroso análisis de datos históricos. A continuación el proceso de construcción:</p>
            <ol>
                <li><strong>Ingesta:</strong> Procesamos transacciones históricas con 17 variables base.</li>
                <li><strong>Preprocesamiento:</strong> Se realizó limpieza de nulos y conversión de tipos temporales (<code>Date</code> a <code>datetime</code>).</li>
                <li><strong>Ingeniería de Variables (KDD):</strong> Se crearon métricas clave como el 'Engagement Score' (Duración sesión * Páginas vistas) y variables temporales.</li>
                <li><strong>Transformación:</strong> Se agregaron los datos por 'Customer_ID', calculando promedios de gasto, recencia de compra y frecuencia.</li>
                <li><strong>Modelado:</strong> Se entrenó un algoritmo de Clasificación Binaria para detectar la probabilidad de abandono (Churn).</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    # Generate Dummy Data for Visualization
    np.random.seed(42)
    dummy_age = np.random.normal(35, 10, 500).astype(int)
    dummy_spent = np.random.exponential(500, 500)
    
    with col1:
        st.markdown("<div class='glass-card'><h4>👥 Distribución de Edad</h4>", unsafe_allow_html=True)
        chart_data_age = pd.DataFrame(dummy_age, columns=["Age"])
        st.bar_chart(chart_data_age["Age"].value_counts().sort_index())
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='glass-card'><h4>💸 Distribución de Gasto (USD)</h4>", unsafe_allow_html=True)
        # Using a simple histogram-like view for spent
        bins = [0, 100, 300, 500, 1000, 5000]
        hist_data = np.histogram(dummy_spent, bins=bins)[0]
        chart_data_spent = pd.DataFrame({"Count": hist_data}, index=[f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)])
        st.bar_chart(chart_data_spent)
        st.markdown("</div>", unsafe_allow_html=True)

# --- TAB 2: PREDICCIÓN ---
with tab2:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.write("### 🎯 Análisis de Riesgo de Abandono")
    st.write("Utiliza los controles del menú lateral para ingresar los datos del cliente y presiona el botón.")
    
    predict_btn = st.button("ANALIZAR RIESGO", use_container_width=True)
    
    if predict_btn:
        with st.spinner("Analizando patrones de comportamiento..."):
            time.sleep(1.5) # Fake loading for dramatic effect
            
            # Prepare input
            input_df = pd.DataFrame({
                'Recency': [recency],
                'Frequency': [frequency],
                'Total_Spent': [monetary],
                'Engagement_Score': [engagement],
                'Rating': [rating],
                'Age': [age]
            })
            
            # Predict
            try:
                prediction = model.predict(input_df)[0]
                proba = model.predict_proba(input_df)[0][1] # Probability of Class 1 (Churn)
            except Exception as e:
                st.error(f"Error en la predicción: {e}")
                prediction = 0
                proba = 0.5
            
            # Display Results
            st.markdown("---")
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                # Gauge Chart (Simulated with progress bar and metrics)
                st.metric("Probabilidad de Churn", f"{proba*100:.1f}%")
                st.progress(proba)
            
            with col_res2:
                if proba > 0.7:
                    st.markdown("""
                        <div style='padding: 20px; border-radius: 10px; background-color: rgba(255, 0, 0, 0.2); border: 2px solid #FF0000; text-align: center;'>
                            <h2 style='color: #FF4444; margin:0;'>⚠️ ALERTA: ALTO RIESGO</h2>
                            <p style='color: white;'>Se recomienda intervenir con descuentos o campaña de fidelización.</p>
                        </div>
                    """, unsafe_allow_html=True)
                elif proba > 0.4:
                     st.markdown("""
                        <div style='padding: 20px; border-radius: 10px; background-color: rgba(255, 165, 0, 0.2); border: 2px solid #FFA500; text-align: center;'>
                            <h2 style='color: #FFA500; margin:0;'>⚠️ RIESGO MODERADO</h2>
                            <p style='color: white;'>Monitorear actividad del usuario.</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div style='padding: 20px; border-radius: 10px; background-color: rgba(0, 255, 0, 0.2); border: 2px solid #00FF00; text-align: center;'>
                            <h2 style='color: #00FF00; margin:0;'>✅ CLIENTE LEAL</h2>
                            <p style='color: white;'>Bajo riesgo de abandono.</p>
                        </div>
                    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style='text-align: center; margin-top: 50px; opacity: 0.5;'>
        <p>E-comChurnnalisys © 2026 | Powered by AI & Streamlit</p>
    </div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    print("\n\n\033[93m⚠️  WARNING: This file is a Streamlit app and cannot be run directly with 'python'.\033[0m")
    print("\033[92m👉 Please run this command instead:\033[0m")
    print("\n    \033[1mpython3 -m streamlit run app.py\033[0m\n")
