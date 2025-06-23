import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from fpdf import FPDF
import io

# ---- CONFIGURACIÓN GENERAL ----
st.set_page_config(page_title="Predicción de Depresión", page_icon="🧠", layout="centered")


st.markdown("""
    <h1 style='text-align: center; color: #34495E;'>🧠 Predicción de Depresión</h1>
    <p style='text-align: center; font-size: 18px;'>Contesta todas las preguntas para obtener una predicción basada en datos</p>
    <hr/>
    """, unsafe_allow_html=True)

# ---- CARGA DEL MODELO ----
with open("/workspaces/Final_Project_DataScient/models/modelo_catboost.pkl", "rb") as file:
    model = pickle.load(file)

# ---- OPCIONES DE ENCODING ----
options = {
    '_EDUCAG': ("Nivel de Educación", {
        "No completó la secundaria": 0, "Se graduó de secundaria": 1,
        "Asistió a colegio técnico o universidad": 2,
        "Se graduó de colegio técnico o universidad": 3
    }),
    'MARITAL': ("Estado civil", {
        "Casado/a": 0, "Divorciado/a": 1, "Viudo/a": 2,
        "Separado/a": 3, "Soltero/a": 4, "En pareja no casada": 5
    }),
    'EMPLOY1': ("Empleo actual", {
        "Empleado/a": 0, "Autónomo/a": 1, "Sin trabajar por 1 año o más": 2,
        "Sin trabajar por menos de 1 año": 3, "Ama de casa": 4,
        "Estudiante": 5, "Jubilado/a": 6, "Incapaz de trabajar": 7
    }),
    '_INCOMG1': ("Ingresos anuales", {
        "Menos de $15,000": 0, "$15,000 a menos de $25,000": 1,
        "$25,000 a menos de $35,000": 2, "$35,000 a menos de $50,000": 3,
        "$50,000 a menos de $100,000": 4, "$100,000 a menos de $200,000": 5,
        "$200,000 o más": 6
    }),
    '_AGEG5YR': ("Rango de edad", {
        "45 a 49 años": 5, "50 a 54 años": 6, "55 a 59 años": 7,
        "60 a 64 años": 8, "65 a 69 años": 9, "70 a 74 años": 10,
        "75 a 79 años": 11, "80 o más": 12
    }),
    'SEXVAR': ("Sexo", {"Hombre": 0, "Mujer": 1}),
    '_ASTHMS1': ("Asma", {"Actual": 1, "Pasada": 2, "Nunca": 3}),
    '_AIDTST4': ("¿Se ha hecho la prueba de VIH?", {"No": 0, "Sí": 1}),
    '_DRDXAR2': ("¿Ha sido diagnosticado con artritis?", {"No": 0, "Sí": 1}),
    '_PHYS14D': ("Días con mala salud física", {"Cero días": 0, "1-13 días": 1, "14 o más días": 2}),
    'DECIDE': ("¿Necesita ayuda para tomar decisiones?", {"No": 0, "Sí": 1}),
    'SDHISOLT': ("¿Se siente aislado socialmente?", {"Siempre/Habitualmente/A veces": 1, "Rara vez/Nunca": 0}),
    'LSATISFY': ("Satisfacción con la vida", {"Muy satisfecho/a": 0, "Satisfecho/a": 1, "Insatisfecho/a": 2, "Muy insatisfecho/a": 3}),
    '_MENT14D': ("Días con mala salud mental", {"Cero días": 0, "1-13 días": 1, "14 o más días": 2}),
    '_BMI5CAT': ("Categoría de IMC", {"Bajo peso": 0, "Peso normal": 1, "Sobrepeso": 2, "Obeso": 3})
}

user_data = {}

# ---- CONFIGURACIÓN DE PESTAÑAS ----
tabs = st.tabs(["👥 Datos Demográficos", "🩺 Salud y Hábitos", "🔍 Predicción"])

# 👥 DATOS DEMOGRÁFICOS
with tabs[0]:
    st.markdown("<h3 style='color: #2980B9;'>👥 Por favor, proporciona tu información demográfica</h3>", unsafe_allow_html=True)
    for var in ['SEXVAR', '_AGEG5YR', 'MARITAL', '_EDUCAG', 'EMPLOY1', '_INCOMG1']:
        user_data[var] = st.selectbox(f"**{options[var][0]}** 👇", list(options[var][1].keys()))

# 🩺 SALUD Y HÁBITOS
with tabs[1]:
    st.markdown("<h3 style='color: #27AE60;'>🩺 Información de Salud y Hábitos</h3>", unsafe_allow_html=True)
    for var in ['_ASTHMS1', '_AIDTST4', '_DRDXAR2']:
        user_data[var] = st.selectbox(f"**{options[var][0]}** 👇", list(options[var][1].keys()))

    st.markdown("""
    **🌡️ Categoría de IMC:**  
    Si no la conoces, calcúlala aquí ➡️ [Calculadora de IMC Externa](https://fundaciondelcorazon.com/prevencion/calculadoras-nutricion/imc.html)  
    """)
    user_data['_BMI5CAT'] = st.selectbox("**Seleccione la categoría correspondiente a su IMC** 👇",
                                          ["Bajo peso", "Peso normal", "Sobrepeso", "Obeso"])
    user_data['_PHYS14D'] = st.selectbox(f"**{options['_PHYS14D'][0]}** 👇", list(options['_PHYS14D'][1].keys()))
    user_data['SLEPTIM1'] = st.slider("🌙 **Horas de sueño** (1-24):", 1, 24, 7)

# 🔍 PREDICCIÓN
with tabs[2]:
    st.markdown("<h3 style='color: #E74C3C;'>🔍 Información de Bienestar Mental</h3>", unsafe_allow_html=True)
    for var in ['DECIDE', 'SDHISOLT', 'LSATISFY', '_MENT14D']:
        user_data[var] = st.selectbox(f"**{options[var][0]}** 👇", list(options[var][1].keys()))

    if st.button("🚀 Realizar Predicción"):
        # Crear DF para predicción
        input_values = {var: options[var][1][user_data[var]] for var in options.keys() if var != '_BMI5CAT'}
        input_values['_BMI5CAT'] = options['_BMI5CAT'][1][user_data['_BMI5CAT']]
        input_values['SLEPTIM1'] = user_data['SLEPTIM1']

        input_df = pd.DataFrame([input_values])
        print(input_df)
        # Realizamos la predicción
        probabilidad = model.predict_proba(input_df)[:, 1][0]
        probabilidad_porcentaje = probabilidad * 100
        resultado = "Posible depresión" if probabilidad >= 0.54 else "No presenta depresión"

        if probabilidad >= 0.54:
            st.error(f"⚠️ Resultado: {resultado}")
            st.markdown("""
                        ### 🌱 Si necesitas apoyo
                        Recuerda que no estás solo/a. Muchas personas y organizaciones pueden ayudarte:
                        - 💚 **Habla con alguien de confianza.**
                        - 👥 Consulta a un especialista en salud mental.
                        - 📞 Llama al [Teléfono de Ayuda contra el Suicidio (024)](https://www.sanidad.gob.es/linea024/)
                        - 🌐 Visita [la Fundación ANAR](https://www.anar.org/) para orientación gratuita y confidencial.
                        - ❤️ Consulta [la AECC](https://www.contraelcancer.es/) para servicios de acompañamiento y apoyo psicológico.

                        Si te encuentras en una situación de crisis, no lo enfrentes solo. ¡Pide ayuda! 🌱
                        """)


        else:
            st.success(f"✅ Resultado: {resultado}")

        st.markdown(f"📊 Probabilidad estimada de depresión: **{probabilidad_porcentaje:.2f}%**.")

        # ---- GRÁFICO DE RESUMEN ----
        labels = ["No Depresión", "Depresión"]
        sizes = [100 - probabilidad_porcentaje, probabilidad_porcentaje]

        fig, ax = plt.subplots()
        ax.bar(labels, sizes, color=["green", "red"])
        ax.set_ylabel("Probabilidad (%)")
        ax.set_title("Probabilidad estimada por el modelo")
        st.pyplot(fig)

        # ---- GENERACIÓN DE PDF ----
        def crear_pdf(resultado, probabilidad_porcentaje, user_data):
            """Crea un PDF con el resultado y las respuestas del usuario."""
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", size=12)

            # Encabezado
            pdf.cell(0, 10, "Resultado de Prediccion de Depresion", ln=True, align='C')
            pdf.ln(5)

            # Resultado
            pdf.cell(0, 10, f"Resultado: {resultado}", ln=True, align='L')
            pdf.cell(0, 10, f"Probabilidad estimada: {probabilidad_porcentaje:.2f}%", ln=True, align='L')
            pdf.ln(5)

            # Respuestas del usuario
            pdf.cell(0, 10, "Respuestas del usuario:", ln=True, align='L')
            for k, v in user_data.items():
                pdf.cell(0, 10, f"{k}: {v}", ln=True, align='L')

            # Exportamos a bytes
            import io
            pdf_buffer = io.BytesIO(pdf.output(dest="S").encode("latin1"))
            pdf_buffer.seek(0)
            return pdf_buffer.getvalue()

        pdf_data = crear_pdf(resultado, probabilidad_porcentaje, user_data)

        st.download_button(
            label="📥 Descargar Resultado en PDF",
            data=pdf_data,
            file_name="resultado_depresion.pdf",
            mime="application/pdf"
        )
