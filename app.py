import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import time

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
    .stNumberInput input, .stSelectbox, .stTextInput, .stSlider {
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: white !important;
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

# --- BACKEND API URL ---
BACKEND_URL = "http://localhost:8000"

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

# --- TAB 2: PREDICCIÓN CON BACKEND ---
with tab2:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.write("### 🎯 Análisis de Riesgo de Abandono (Backend Powered)")
    
    mode = st.radio("Selecciona el modo de predicción:", ["Predicción Individual", "Predicción por Lote (CSV/Excel)"], horizontal=True)
    st.markdown("---")
    
    if mode == "Predicción Individual":
        st.write("#### 👤 Ingrese los datos del cliente:")
        
        with st.form("single_predict_form"):
            c1, c2, c3 = st.columns(3)
            
            with c1:
                tenure = st.number_input("Tenure (Meses)", min_value=0, value=10)
                city_tier = st.selectbox("City Tier", [1, 2, 3])
                warehouse_to_home = st.number_input("Warehouse To Home (km)", min_value=0.0, value=15.0)
                hour_spend_on_app = st.number_input("Hour Spend On App", min_value=0.0, value=3.0)
                number_of_device_registered = st.number_input("Devices Registered", min_value=1, step=1, value=4)
                satisfaction_score = st.slider("Satisfaction Score", 1, 5, 3)

            with c2:
                number_of_address = st.number_input("Number Of Address", min_value=1, step=1, value=5)
                complain = st.selectbox("Complain (Quejas)", options=[0, 1], format_func=lambda x: "Sí (1)" if x == 1 else "No (0)")
                order_amount_hike = st.number_input("Order Amount Hike (%)", min_value=0.0, value=15.0)
                coupon_used = st.number_input("Coupon Used", min_value=0, step=1, value=2)
                order_count = st.number_input("Order Count", min_value=0, step=1, value=3)
                day_since_last_order = st.number_input("Days Since Last Order", min_value=0.0, value=5.0)

            with c3:
                cashback_amount = st.number_input("Cashback Amount", min_value=0.0, value=180.0)
                preferred_login_device = st.selectbox("Preferred Login Device", options=[0, 1, 2], format_func=lambda x: ["Computer (0)", "Mobile Phone (1)", "Phone (2)"][x] if x<3 else x)
                preferred_payment_mode = st.selectbox("Preferred Payment Mode", options=[0, 1, 2, 3, 4, 5, 6], format_func=lambda x: ["CC (0)", "COD (1)", "Cash on Delivery (2)", "Credit Card (3)", "Debit Card (4)", "E Wallet (5)", "UPI (6)"][x] if x<7 else x)
                gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Female (0)" if x == 0 else "Male (1)")
                prefered_order_cat = st.selectbox("Preferred Order Cat", options=[0, 1, 2, 3, 4, 5], format_func=lambda x: ["Fashion (0)", "Grocery (1)", "Laptop & Accessory (2)", "Mobile (3)", "Mobile Phone (4)", "Others (5)"][x] if x<6 else x)
                marital_status = st.selectbox("Marital Status", options=[0, 1, 2], format_func=lambda x: ["Divorced (0)", "Married (1)", "Single (2)"][x] if x<3 else x)
            
            submit_btn = st.form_submit_button("PREDECIR INDIVIDUAL", use_container_width=True)
        
        if submit_btn:
            payload = {
                "tenure": tenure,
                "city_tier": city_tier,
                "warehouse_to_home": warehouse_to_home,
                "hour_spend_on_app": hour_spend_on_app,
                "number_of_device_registered": number_of_device_registered,
                "satisfaction_score": satisfaction_score,
                "number_of_address": number_of_address,
                "complain": complain,
                "order_amount_hike_from_last_year": order_amount_hike,
                "coupon_used": coupon_used,
                "order_count": order_count,
                "day_since_last_order": day_since_last_order,
                "cashback_amount": cashback_amount,
                "preferred_login_device": preferred_login_device,
                "preferred_payment_mode": preferred_payment_mode,
                "gender": gender,
                "prefered_order_cat": prefered_order_cat,
                "marital_status": marital_status
            }
            
            with st.spinner("Conectando con el Backend..."):
                try:
                    response = requests.post(f"{BACKEND_URL}/predict", json=payload)
                    response.raise_for_status()
                    result = response.json()
                    
                    # Logica flexible para respuesta del backend
                     # Probabilidad
                    if "probability" in result:
                        prob = result["probability"]
                    elif "churn_probability" in result:
                        prob = result["churn_probability"]
                    else:
                        # Fallback si no viene la probabilidad
                        prob = 0.9 if result.get("prediction", 0) == 1 else 0.1
                        
                    is_churn = result.get("prediction", 0)
                    
                    st.markdown("### 📊 Resultado de la Predicción")
                    
                    c_res1, c_res2 = st.columns([1, 2])
                    with c_res1:
                        st.metric("Probabilidad de Churn", f"{prob*100:.1f}%")
                        st.progress(prob)
                        
                    with c_res2:
                        if prob > 0.7:
                            st.error(f"⚠️ RIESGO ALTO (Churn: {'Sí' if is_churn else 'No'}) - El cliente está en peligro de abandonar.")
                        elif prob > 0.4:
                            st.warning(f"⚠️ RIESGO MEDIO (Churn: {'Sí' if is_churn else 'No'}) - Monitorear cliente.")
                        else:
                            st.success(f"✅ RIESGO BAJO (Churn: {'Sí' if is_churn else 'No'}) - Cliente seguro.")
                            
                except requests.exceptions.ConnectionError:
                    st.error(f"❌ Error de conexión: No se pudo conectar a {BACKEND_URL}. Asegúrate que el backend esté corriendo.")
                except Exception as e:
                    st.error(f"❌ Error al procesar la solicitud: {e}")

    elif mode == "Predicción por Lote (CSV/Excel)":
        st.write("#### 📂 Carga masiva de clientes")
        st.info("Sube un archivo con las columnas requeridas (nombres en snake_case como en la API).")
        
        uploaded_file = st.file_uploader("Arrastra tu archivo aquí", type=["csv", "xlsx", "json"])
        
        if uploaded_file:
            try:
                # Lectura del archivo
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    df = pd.read_json(uploaded_file)
                
                st.write(f"Previsualización ({len(df)} registros):")
                st.dataframe(df.head())
                
                if st.button("PROCESAR LOTE", use_container_width=True):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Preparar payload
                    # Asumimos que el CSV ya tiene las columnas con los nombres correctos
                    # Convertimos a lista de dicts
                    customers_list = df.to_dict(orient="records")
                    payload = {"customers": customers_list}
                    
                    status_text.text("Enviando datos al backend...")
                    progress_bar.progress(30)
                    
                    try:
                        response = requests.post(f"{BACKEND_URL}/predict-batch", json=payload)
                        response.raise_for_status()
                        progress_bar.progress(80)
                        
                        api_response = response.json()
                        # Se espera algo como: {"predictions": [0, 1, 0, ...]} o una lista directa
                        
                        if isinstance(api_response, dict) and "predictions" in api_response:
                            predictions = api_response["predictions"]
                        elif isinstance(api_response, list):
                            predictions = api_response
                        else:
                             predictions = []
                             st.error("Formato de respuesta desconocido")

                        if len(predictions) == len(df):
                            # Manejar si predictions es lista de objetos o lista de ints
                            if len(predictions) > 0 and isinstance(predictions[0], dict):
                                # Logic to find the probability key dynamically
                                sample_pred = predictions[0]
                                prob_key = "probability"
                                if "churn_probability" in sample_pred:
                                    prob_key = "churn_probability"
                                elif "prob" in sample_pred:
                                    prob_key = "prob"
                                
                                df['Churn_Prediction'] = [p.get('prediction', p.get('churn', 0)) for p in predictions]
                                df['Churn_Probability'] = [p.get(prob_key, 0.0) for p in predictions]
                            else:
                                df['Churn_Prediction'] = predictions
                                # Dummy prob si no viene
                                df['Churn_Probability'] = df['Churn_Prediction'].apply(lambda x: 0.9 if x==1 else 0.1)
                            
                            progress_bar.progress(100)
                            status_text.text("¡Completado!")
                            
                            st.success("✅ Predicciones recibidas exitosamente.")
                            
                            # Función para colorear
                            def color_risk(val):
                                if val > 0.7:
                                    return 'background-color: rgba(255, 0, 0, 0.5); color: white'
                                elif val > 0.4:
                                    return 'background-color: rgba(255, 165, 0, 0.5); color: white'
                                else:
                                    return 'background-color: rgba(0, 255, 0, 0.3); color: white'

                            # Formatear y mostrar
                            st.dataframe(
                                df.style.map(color_risk, subset=['Churn_Probability'])
                                .format({'Churn_Probability': '{:.1%}'}),
                                use_container_width=True
                            )
                            
                            # Botón de descarga
                            csv_data = df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download Predictions as CSV",
                                data=csv_data,
                                file_name='churn_predictions.csv',
                                mime='text/csv',
                            )
                        else:
                            st.warning("⚠️ La cantidad de predicciones no coincide con los registros enviados.")
                            
                    except Exception as e:
                        st.error(f"Error en la petición: {e}")
                        
            except Exception as e:
                st.error(f"Error leyendo el archivo: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style='text-align: center; margin-top: 50px; opacity: 0.5;'>
        <p>E-comChurnnalisys © 2026 | Powered by AI & Streamlit</p>
    </div>
""", unsafe_allow_html=True)
