import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from fpdf import FPDF
import io
import os
import numpy as np

# --- Configuración inicial ---
st.set_page_config(page_title="Evaluador de Riesgo de Depresión ☠️", page_icon="🚨", layout="centered")
pages = ["👩🏻‍🦰👨🏻‍🦰 Datos Generales", "🩺 Salud y Hábitos", "📝 PHQ-9", "🔍 Resultado"]

# --- Estilos ---
st.markdown("""
    <style>
    .stApp { background-color: #f2f6fc; font-family: 'Segoe UI', sans-serif; }
    .card { background-color: white; padding: 2rem; border-radius: 15px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.05); margin-bottom: 2rem; }
    .titulo-azul { font-size: 42px; font-weight: 700; color: #1a5276; text-align: center; margin-bottom: 1rem; }
    .subtitulo-negro { font-size: 24px; font-weight: 600; color: #212121; text-align: center; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

# --- Mostrar título dinámico con emoji ---
st.markdown(f'<div class="titulo-azul">{pages[st.session_state.get("page", 0)]}</div>', unsafe_allow_html=True)

# --- Aviso ---
st.markdown("""
<div class="card">
<b>🔒 Aviso importante:</b>
Esta herramienta es solo orientativa y no sustituye un diagnóstico médico profesional.
Los datos proporcionados son confidenciales. Si tienes síntomas persistentes, consulta a un profesional de la salud.
</div>
""", unsafe_allow_html=True)

# --- Cargar modelo ---
@st.cache_resource
def cargar_modelo():
    modelo_path = os.path.join(os.path.dirname(__file__), "modelo_catboost.pkl")
    with open(modelo_path, "rb") as file:
        return pickle.load(file)

model = cargar_modelo()

# --- Diccionarios ---
bin_map = {"No": 0, "Sí": 1}
sex_map = {"Hombre": 0, "Mujer": 1}
edad_map = {"18-24": 0, "25-29": 1, "30-34": 2, "35-39": 3, "40-44": 4, "45-49": 5, "50-54": 6,
    "55-59": 7, "60-64": 8, "65-69": 9, "70-74": 10, "75-79": 11, "80+": 12}
educ_map = {"No completó secundaria": 0, "Graduado secundaria": 1, "Asistió a universidad": 2, "Graduado universidad": 3}
income_map = {"<15k": 0, "15-25k": 1, "25-35k": 2, "35-50k": 3, "50-100k": 4, "100-200k": 5, "200k+": 6}
bmi_map = {"Bajo peso": 0, "Normal": 1, "Sobrepeso": 2, "Obeso": 3}
phys_map = {"Cero días": 0, "1-13 días": 1, "14 o más días": 2}
satisfy_map = {"Muy satisfecho/a": 0, "Satisfecho/a": 1, "Insatisfecho/a": 2, "Muy insatisfecho/a": 3}
employ_map = {"Empleado": 1, "Desempleado": 2, "Jubilado": 3, "Estudiante": 4,"Otro": 5}
marital_map = {"Casado/a": 1, "Soltero/a": 2, "Separado/a": 3, "Viudo/a": 4}

# --- Inicialización de estado ---
if "page" not in st.session_state:
    st.session_state.page = 0
if "data" not in st.session_state:
    st.session_state.data = {}
if "phq_answers" not in st.session_state:
    st.session_state.phq_answers = [None] * 9
data = st.session_state.data

# --- Página 0: Datos Generales ---
if st.session_state.page == 0:
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        data['SEXVAR'] = sex_map[st.radio("Sexo", list(sex_map.keys()))]
        data['_AGEG5YR'] = edad_map[st.selectbox("Edad", list(edad_map.keys()))]
        data['_EDUCAG'] = educ_map[st.selectbox("Nivel educativo", list(educ_map.keys()))]
        data['_INCOMG1'] = income_map[st.selectbox("Nivel de ingresos", list(income_map.keys()))]
        data['EMPLOY1'] = employ_map[st.selectbox("Situación laboral", list(employ_map.keys()))]
        data['MARITAL'] = marital_map[st.selectbox("Estado civil", list(marital_map.keys()))]
        st.markdown('</div>', unsafe_allow_html=True)

# --- Página 1: Salud y Hábitos ---
elif st.session_state.page == 1:
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        with st.form(key="form_pagina1"):
            data['SLEPTIM1'] = st.slider("Horas de sueño promedio", 0, 24, 7)
            data['DECIDE'] = bin_map[st.radio("¿Tiene dificultades para tomar decisiones?", list(bin_map.keys()))]
            data['SDHISOLT'] = bin_map[st.radio("¿Se siente aislado frecuentemente?", list(bin_map.keys()))]
            data['_DRDXAR2'] = bin_map[st.radio("¿Diagnóstico profesional de depresión?", list(bin_map.keys()))]
            data['LSATISFY'] = satisfy_map[st.selectbox("Satisfacción con la vida", list(satisfy_map.keys()))]
            data['_PHYS14D'] = phys_map[st.radio("Días con problemas físicos en últimas 2 semanas", list(phys_map.keys()))]
            data['_MENT14D'] = phys_map[st.radio("Días con problemas mentales en últimas 2 semanas", list(phys_map.keys()))]
            data['_AIDTST4'] = bin_map[st.radio("¿Ha hecho prueba de VIH?", list(bin_map.keys()))]
            data['_ASTHMS1'] = bin_map[st.radio("¿Tiene diagnóstico de asma?", list(bin_map.keys()))]

            peso = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0, value=70.0)
            altura = st.number_input("Altura (cm)", min_value=100.0, max_value=220.0, value=170.0)
            imc = peso / ((altura / 100) ** 2)
            imc_cat = "Bajo peso" if imc < 18.5 else "Normal" if imc < 25 else "Sobrepeso" if imc < 30 else "Obeso"
            data['_BMI5CAT'] = bmi_map[imc_cat]
            data['peso'] = peso
            data['altura'] = altura
            st.markdown(f"**IMC:** {imc:.1f} - {imc_cat}")
            st.session_state.imc = f"{imc:.1f} - {imc_cat}"

            submitted = st.form_submit_button("Guardar y continuar ➡️")
            if submitted:
                if all(k in data for k in ['SLEPTIM1', 'DECIDE', 'SDHISOLT', '_DRDXAR2', 'LSATISFY', '_PHYS14D', '_MENT14D', '_BMI5CAT', '_AIDTST4', '_ASTHMS1']):
                    st.session_state.page += 1
                else:
                    st.warning("⚠️ Por favor completa todos los campos antes de continuar.")
        st.markdown('</div>', unsafe_allow_html=True)

# --- Página 2: PHQ-9 ---
elif st.session_state.page == 2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("Contesta cuántas veces te han afectado los siguientes síntomas en las últimas 2 semanas:")

    preguntas = [
        "Poco interés o placer en hacer cosas",
        "Sentirse deprimido o sin esperanza",
        "Dificultad para dormir o dormir demasiado",
        "Sentirse cansado o con poca energía",
        "Falta de apetito o comer en exceso",
        "Sentirse mal consigo mismo",
        "Dificultad para concentrarse",
        "Moverse o hablar más lento o agitado de lo normal",
        "Pensamientos de autolesión o muerte"
    ]

    opciones = ["Nunca (0)", "Varios días (1)", "Más de la mitad de los días (2)", "Casi todos los días (3)"]

    for i, pregunta in enumerate(preguntas):
        valor_prev = st.session_state.phq_answers[i]
        index_pred = valor_prev if valor_prev is not None else 0
        resp = st.radio(f"{i+1}. {pregunta}", opciones, key=f"phq_{i}", index=index_pred)
        st.session_state.phq_answers[i] = int(resp.split("(")[1][0])

    st.session_state.phq9_score = sum(st.session_state.phq_answers)
    st.markdown(f"### 🧮 Puntaje total PHQ-9: **{st.session_state.phq9_score} / 27**")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Página 3: Resultado ---
elif st.session_state.page == 3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="subtitulo-negro">📊 Resultados</div>', unsafe_allow_html=True)

    input_df = pd.DataFrame([data])
    for col in model.feature_names_:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model.feature_names_]

    prob = model.predict_proba(input_df)[0][1]
    prob_pct = prob * 100
    score = st.session_state.phq9_score
    alerta = prob >= 0.54 or score >= 10

    st.success(f"**Probabilidad del modelo:** {prob_pct:.2f}%")
    st.info(f"**Puntaje PHQ-9:** {score} / 27")

    if alerta:
        st.error("⚠️ Indicios de riesgo de depresión detectados.")
    else:
        st.success("✅ Sin indicios claros de depresión.")

    # --- Resultado visual ---
    st.markdown("### Prediccion")
    if prob >= 0.70 or score >= 15:
        st.markdown("<div style='font-size:100px; text-align:center;'>🧨</div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align:center; font-size:20px; color:#922B21; font-weight:bold;'>Alto riesgo de depresión</div>", unsafe_allow_html=True)
    elif prob >= 0.54 or score >= 10:
        st.markdown("<div style='font-size:100px; text-align:center;'>😟</div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align:center; font-size:20px; color:#B9770E; font-weight:bold;'>Posibles signos de depresión</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='font-size:100px; text-align:center;'>😊</div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align:center; font-size:20px; color:#239B56; font-weight:bold;'>Sin indicios de depresión</div>", unsafe_allow_html=True)

    # --- Recomendaciones si hay alerta ---
    if alerta:
        st.markdown("""---""")
        st.markdown('<div class="subtitulo-azul">🧠 ¿Qué puedes hacer?</div>', unsafe_allow_html=True)
        st.markdown("""
        Si tú o alguien que conoces presenta síntomas de depresión, es fundamental buscar ayuda profesional.

        **Recomendaciones:**
        - Habla con un **psicólogo** o **psiquiatra** titulado.
        - Contacta con el **médico de cabecera** para una evaluación clínica.
        - Comenta tus síntomas con alguien de confianza.
        - Llama a líneas de ayuda emocional o de crisis.

        **Recursos útiles:**
        - 📞 [Teléfono de la Esperanza (España)](https://telefonodelaesperanza.org) – 717 003 717
        - 🌐 [Colegio Oficial de Psicólogos](https://www.cop.es/)
        - 🚨 Si estás en peligro inmediato, llama al **112**
        
        > *Tu bienestar es importante. No estás solo/a, recuerda los momentos difíciles también pasan; ten paciencia contigo mismo.* 💙
        """)

    # --- PDF ---
    def crear_pdf():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.cell(0, 10, "Resultado de Evaluación", ln=True)
        pdf.cell(0, 10, f"Probabilidad: {prob_pct:.2f}%", ln=True)
        pdf.cell(0, 10, f"Puntaje PHQ-9: {score}", ln=True)
        pdf.cell(0, 10, f"IMC: {st.session_state.get('imc', 'No disponible')}", ln=True)
        for k, v in data.items():
            pdf.cell(0, 8, f"{k}: {v}", ln=True)
        buffer = io.BytesIO(pdf.output(dest='S').encode('latin1'))
        buffer.seek(0)
        return buffer.getvalue()

    st.download_button("📄 Descargar PDF", crear_pdf(), file_name="resultado_depresion.pdf", mime="application/pdf")

    st.markdown('</div>', unsafe_allow_html=True)
# --- Navegación ---
def validar_pagina_0(): return all(k in data for k in ['SEXVAR', '_AGEG5YR', '_EDUCAG', '_INCOMG1', 'EMPLOY1', 'MARITAL'])
def validar_pagina_2(): return None not in st.session_state.phq_answers

col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    if st.button("⬅️ Anterior") and st.session_state.page > 0:
        st.session_state.page -= 1
with col3:
    if st.session_state.page != 1 and st.button("Siguiente ➡️"):
        validadores = [validar_pagina_0, None, validar_pagina_2]
        if st.session_state.page < 3:
            if validadores[st.session_state.page]() if validadores[st.session_state.page] else True:
                st.session_state.page += 1
            else:
                st.warning("⚠️ Por favor completa todos los campos antes de continuar.")

# --- Progreso ---
progress_value = min((st.session_state.page + 1) / len(pages), 1.0)
st.progress(progress_value, text=f"Progreso: Página {st.session_state.page + 1} de {len(pages)}")