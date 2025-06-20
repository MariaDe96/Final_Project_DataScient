import streamlit as st
import pandas as pd
import numpy as np

from catboost import CatBoostClassifier
import joblib

# Cargar modelo
@st.cache_resource
def cargar_modelo():
    # modelo = joblib.load('/workspaces/Final_Project_DataScient/app/modelo_catboost.pkl')
    return None # modelo

modelo = cargar_modelo()

# --- 1. Datos Demográficos y Generales ---

AGEG5YR_data = {
    "pregunta": "¿A cuál de las siguientes categorías de edad pertenece?",
    "opciones": {
        "De 18 a 24 años": 0,
        "De 25 a 29 años": 1,
        "De 30 a 34 años": 2,
        "De 35 a 39 años": 3,
        "De 40 a 44 años": 4,
        "De 45 a 49 años": 5,
        "De 50 a 54 años": 6,
        "De 55 a 59 años": 7,
        "De 60 a 64 años": 8,
        "De 65 a 69 años": 9,
        "De 70 a 74 años": 10,
        "De 75 a 79 años": 11,
        "80 años o más": 12
    }
}

SEXVAR_data = {
    "pregunta": "¿Cuál es tu sexo (asignado al nacer)?",
    "opciones": {
        "Hombre": 0,
        "Mujer": 1
    }
}

EDUCAG_data = {
    "pregunta": "¿Cuál es el nivel educativo más alto que has alcanzado?",
    "opciones": {
        "No se graduó de la escuela secundaria": 0,
        "Se graduó de la escuela secundaria": 1,
        "Asistió a la Universidad o Escuela Técnica": 2,
        "Graduado de Colegio o Escuela Técnica": 3
    }
}

MARITAL_data = {
    "pregunta": "¿Cuál es tu estado civil actual?",
    "opciones": {
        "Casado/a": 0,
        "Divorciado/a": 1,
        "Viudo/a": 2,
        "Separado/a legalmente": 3,
        "Soltero/a": 4,
        "En una relación de pareja de hecho o unión libre": 5
    }
}

HHADULT_data = {
    "pregunta": "¿Cuántos miembros de su hogar, incluido usted mismo, tienen 18 años de edad o más?",
    "opciones": {
        "type": "numerical_range",
        "min": 1,
        "max": 20
    }
}

CHLDCNT_data = {
    "pregunta": "¿Cuántos niños menores de 18 años viven en su hogar?",
    "opciones": {
        "No hay": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "Hay 5 o más": 5
    }
}

RENTHOM1_data = {
    "pregunta": "¿Es usted propietario o alquila su vivienda?",
    "opciones": {
        "Soy propietario/a": 0,
        "Alquilo": 1,
        "Otra situación": 2
    }
}

EMPLOY1_data = {
    "pregunta": "¿Cuál es tu situación laboral actual?",
    "opciones": {
        "Empleado/a por cuenta ajena": 0,
        "Trabajador/a autónomo/a": 1,
        "En paro (1 año o más)": 2,
        "En paro (menos de 1 año)": 3,
        "Persona dedicada a las tareas del hogar": 4,
        "Estudiante": 5,
        "Jubilado/a": 6,
        "Incapacitado/a para trabajar": 7
    }
}

INCOMG1_data = {
    "pregunta": "¿En cuál de las siguientes categorías se encuentra el ingreso total anual de su hogar (antes de impuestos)?",
    "opciones": {
        "Menos 15000": 0,
        "De 15.000 a 25.000": 1,
        "De 25.000 a 35.000": 2,
        "De 35.000 a 50.000": 3,
        "De 50.000 a 100.000": 4,
        "De 100.000 a 200.000": 5,
        "Más de 200.000": 6
    }
}

BMI5CAT_data = {
    "pregunta": "Según su Índice de Masa Corporal (IMC), ¿a cuál de las siguientes categorías pertenece?",
    "opciones": {
        "Bajo peso": 0,
        "Peso normal": 1,
        "Sobrepeso": 2,
        "Obesidad": 3
    }
}

# --- 2. Hábitos y Percepción General de Salud ---

SLEPTIM1_data = {
    "pregunta": "De media, ¿cuántas horas duermes en un periodo de 24 horas?",
    "opciones": {
        "type": "numerical_range",
        "min": 1,
        "max": 24
    }
}

LSATISFY_data = {
    "pregunta": "En general, ¿qué tan satisfecho/a se siente con su vida?",
    "opciones": {
        "Muy satisfecho/a": 0,
        "Satisfecho/a": 1,
        "Insatisfecho/a": 2,
        "Muy insatisfecho/a": 3
    }
}

PHYS14D_data = {
    "pregunta": "Durante los últimos 30 días, ¿cuántos días su salud física no fue buena?",
    "opciones": {
        "0 dias": 0,
        "1 a 13 días": 1,
        "Más de 13 días": 2
    }
}

# --- 3. Limitaciones Funcionales y Historial Médico ---

DECIDE_data = {
    "pregunta": "Debido a una condición física, mental o emocional, ¿tiene usted serias dificultades para concentrarse, recordar o tomar decisiones?",
    "opciones": {
        "No": 0,
        "Sí": 1
    }
}

DIFFALON_data = {
    "pregunta": "Debido a una condición física, mental o emocional, ¿tiene usted dificultad para hacer recados solo, como visitar al consultorio del médico o ir de compras?",
    "opciones": {
        "No": 0,
        "Sí": 1
    }
}

MICHD_data = {
    "pregunta": "¿Alguna vez le han diagnosticado una enfermedad coronaria (EC) o ha sufrido un infarto de miocardio (IM)?",
    "opciones": {
        "No": 0,
        "Sí": 1
    }
}

# --- 4. Acceso a Recursos y Comportamientos de Riesgo ---

MEDCOST1_data = {
    "pregunta": "¿Hubo alguna ocasión en los últimos 12 meses en la que usted necesitó ir al médico pero no pudo porque no podía costearlo?",
    "opciones": {
        "No": 0,
        "Sí": 1
    }
}

HIVRISK5_data = {
    "pregunta": """"Voy a leerle una lista. Por favor, dígame 'Sí' o 'No' si alguna de las siguientes situaciones se aplica a usted en los últimos 12 meses:
* ¿Se ha inyectado alguna droga que no le haya sido recetada?
* ¿Ha recibido tratamiento por una enfermedad de transmisión sexual (ETS)?
* ¿Ha dado o recibido dinero o drogas a cambio de sexo?""",
    "opciones": {
        "No": 0,
        "Sí": 1
    }
}

# --- 5. Bienestar Emocional y Mental Directo ---

SDHISOLT_data = {
    "pregunta": "¿Con qué frecuencia se siente socialmente aislado de los demás?",
    "opciones": {
        "Siempre": 1,
        "Usualmente": 1,
        "Algunas veces": 1,
        "Rara vez": 0,
        "Nunca": 0
    }
}

MENT14D_data = {
    "pregunta": "En los últimos 30 días, ¿cuántos días se ha sentido con un estado de salud no bueno (física o mentalmente)",
    "opciones": {
        "0 dias": 0,
        "1 a 13 días": 1,
        "Más de 13 días": 2
    }
}


# Definimos el orden de las columnas para entregarselo al modelo.
model_feature_order = [
    "_MENT14D",
    "SDHISOLT",
    "LSATISFY",
    "_PHYS14D",
    "SLEPTIM1",
    "_INCOMG1",
    "_BMI5CAT",
    "_AGEG5YR",
    "_EDUCAG",
    "DECIDE",
    "SEXVAR",
    "HHADULT",
    "DIFFALON",
    "_MICHD",
    "_CHLDCNT",
    "MEDCOST1",
    "HIVRISK5",
    "MARITAL",
    "RENTHOM1",
    "EMPLOY1"
]

# Unificar los diccionarios de preguntas para fácil acceso
all_question_definitions = {
    "SDHISOLT": SDHISOLT_data,
    "_MENT14D": MENT14D_data,
    "LSATISFY": LSATISFY_data,
    "_PHYS14D": PHYS14D_data,
    "SLEPTIM1": SLEPTIM1_data,
    "_INCOMG1": INCOMG1_data,
    "_BMI5CAT": BMI5CAT_data,
    "_AGEG5YR": AGEG5YR_data,
    "_EDUCAG": EDUCAG_data,
    "DECIDE": DECIDE_data,
    "SEXVAR": SEXVAR_data,
    "HHADULT": HHADULT_data,
    "DIFFALON": DIFFALON_data,
    "_MICHD": MICHD_data,
    "_CHLDCNT": CHLDCNT_data,
    "MEDCOST1": MEDCOST1_data,
    "HIVRISK5": HIVRISK5_data,
    "MARITAL": MARITAL_data,
    "RENTHOM1": RENTHOM1_data,
    "EMPLOY1": EMPLOY1_data,
}
def run_survey():
    st.title("Encuesta para la Predicción de Depresión")
    st.write("Por favor, responda las siguientes preguntas para ayudarnos a recopilar información.")
    st.write("Sus respuestas son confidenciales y se utilizarán para un proyecto de investigación.")

    user_responses = {}

        # --- 1. Datos Demográficos y Generales --- 
        #  Edad
    st.subheader(AGEG5YR_data["pregunta"])
    options_AGEG5YR = [" "] + list(AGEG5YR_data["opciones"].keys())
    selected_option_AGEG5YR = st.selectbox(
        "Selecciona una opción:", # Etiqueta opcional para el selectbox
        options_AGEG5YR,
        key="AGEG5YR_q"
    )

    # Aquí es donde debes manejar la asignación a user_responses con una validación
    if selected_option_AGEG5YR != " ":
        # Solo asigna si el usuario seleccionó una opción válida (no el placeholder)
        user_responses["AGEG5YR"] = AGEG5YR_data["opciones"][selected_option_AGEG5YR]
    else:
        # Si el usuario no ha seleccionado nada o dejó el placeholder,
        # asigna None para indicar que no hay una respuesta válida
        user_responses["AGEG5YR"] = None
    
        # Sexo

    st.subheader(SEXVAR_data["pregunta"])
    options_SEXVAR = list(SEXVAR_data["opciones"].keys())
    selected_option_SEXVAR = st.radio(
        "Selecciona una opción:", # Etiqueta opcional para el selectbox
        options_SEXVAR,
        key="SEXVAR_q"
    )

    user_responses["SEXVAR"] = SEXVAR_data["opciones"][selected_option_SEXVAR]

    # Nivel educativo

    st.subheader(EDUCAG_data["pregunta"])
    options_EDUCAG = [" "] + list(EDUCAG_data["opciones"].keys())
    selected_option_EDUCAG = st.selectbox(
        "Selecciona una opción:", # Etiqueta opcional para el selectbox
        options_EDUCAG,
        key="EDUCAG_q"
    )

    # Aquí es donde debes manejar la asignación a user_responses con una validación
    if selected_option_EDUCAG != " ":
        # Solo asigna si el usuario seleccionó una opción válida (no el placeholder)
        user_responses["EDUCAG"] = EDUCAG_data["opciones"][selected_option_EDUCAG]
    else:
        # Si el usuario no ha seleccionado nada o dejó el placeholder,
        # asigna None para indicar que no hay una respuesta válida
        user_responses["EDUCAG"] = None

    # Estado civil

    st.subheader(MARITAL_data["pregunta"])
    options_MARITAL = [" "] + list(MARITAL_data["opciones"].keys())
    selected_option_MARITAL = st.selectbox(
        "Selecciona una opción:", # Etiqueta opcional para el selectbox
        options_MARITAL,
        key="MARITAL_q"
    )

    # Aquí es donde debes manejar la asignación a user_responses con una validación
    if selected_option_MARITAL != " ":
        # Solo asigna si el usuario seleccionó una opción válida (no el placeholder)
        user_responses["MARITAL"] = MARITAL_data["opciones"][selected_option_MARITAL]
    else:
        # Si el usuario no ha seleccionado nada o dejó el placeholder,
        # asigna None para indicar que no hay una respuesta válida
        user_responses["MARITAL"] = None


    # Miembros del hogar

    st.subheader(HHADULT_data["pregunta"])
    if HHADULT_data["opciones"]["type"] == "numerical_range":
        num_miembros = st.number_input(
            "Ingrese el número de personas:",
            min_value=HHADULT_data["opciones"]["min"],
            max_value=HHADULT_data["opciones"]["max"],
            value=2, # Valor por defecto
            step=1,
            key="HHADULT_q"
        )
        user_responses["HHADULT"] = num_miembros


    # Número de niños en el hogar

    st.subheader(CHLDCNT_data["pregunta"])

    options_CHLDCNT = list(CHLDCNT_data["opciones"].keys())

    selected_option_chldcnt_label = st.select_slider(
        "", # No necesitamos una etiqueta grande ya que el subheader ya es la pregunta
        options=options_CHLDCNT,
        key="chldcnt_q"
    )

    user_responses["CHLDCNT"] = CHLDCNT_data["opciones"][selected_option_chldcnt_label]


        # Situación de la casa        

    st.subheader(RENTHOM1_data["pregunta"])
    options_RENTHOM1 = [" "] + list(RENTHOM1_data["opciones"].keys())
    selected_option_RENTHOM1 = st.selectbox(
        "Selecciona una opción:", # Etiqueta opcional para el selectbox
        options_RENTHOM1,
        key="RENTHOM1_q"
    )

    # Aquí es donde debes manejar la asignación a user_responses con una validación
    if selected_option_RENTHOM1 != " ":
        # Solo asigna si el usuario seleccionó una opción válida (no el placeholder)
        user_responses["RENTHOM1"] = RENTHOM1_data["opciones"][selected_option_RENTHOM1]
    else:
        # Si el usuario no ha seleccionado nada o dejó el placeholder,
        # asigna None para indicar que no hay una respuesta válida
        user_responses["RENTHOM1"] = None


        # Situación laboral      

    st.subheader(EMPLOY1_data["pregunta"])
    options_EMPLOY1 = [" "] + list(EMPLOY1_data["opciones"].keys())
    selected_option_EMPLOY1 = st.selectbox(
        "Selecciona una opción:", # Etiqueta opcional para el selectbox
        options_EMPLOY1,
        key="EMPLOY1_q"
    )

    # Aquí es donde debes manejar la asignación a user_responses con una validación
    if selected_option_EMPLOY1 != " ":
        # Solo asigna si el usuario seleccionó una opción válida (no el placeholder)
        user_responses["EMPLOY1"] = EMPLOY1_data["opciones"][selected_option_EMPLOY1]
    else:
        # Si el usuario no ha seleccionado nada o dejó el placeholder,
        # asigna None para indicar que no hay una respuesta válida
        user_responses["EMPLOY1"] = None


        # Ingresos   

    st.subheader(INCOMG1_data["pregunta"])
    options_INCOMG1 = [" "] + list(INCOMG1_data["opciones"].keys())
    selected_option_INCOMG1 = st.selectbox(
        "Selecciona una opción:", # Etiqueta opcional para el selectbox
        options_INCOMG1,
        key="INCOMG1_q"
    )

    # Aquí es donde debes manejar la asignación a user_responses con una validación
    if selected_option_INCOMG1 != " ":
        # Solo asigna si el usuario seleccionó una opción válida (no el placeholder)
        user_responses["INCOMG1"] = INCOMG1_data["opciones"][selected_option_INCOMG1]
    else:
        # Si el usuario no ha seleccionado nada o dejó el placeholder,
        # asigna None para indicar que no hay una respuesta válida
        user_responses["INCOMG1"] = None


        # Indice de masa corporal  

    st.subheader(BMI5CAT_data["pregunta"])
    options_BMI5CAT = [" "] + list(BMI5CAT_data["opciones"].keys())
    selected_option_BMI5CAT = st.selectbox(
        "Selecciona una opción:", # Etiqueta opcional para el selectbox
        options_BMI5CAT,
        key="BMI5CAT_q"
    )

    # Aquí es donde debes manejar la asignación a user_responses con una validación
    if selected_option_BMI5CAT != " ":
        # Solo asigna si el usuario seleccionó una opción válida (no el placeholder)
        user_responses["BMI5CAT"] = BMI5CAT_data["opciones"][selected_option_BMI5CAT]
    else:
        # Si el usuario no ha seleccionado nada o dejó el placeholder,
        # asigna None para indicar que no hay una respuesta válida
        user_responses["BMI5CAT"] = None


    # --- 2. Hábitos y Percepción General de Salud ---

    # Horas de sueño

    st.subheader(SLEPTIM1_data["pregunta"])
    if SLEPTIM1_data["opciones"]["type"] == "numerical_range":
        sleep_hours = st.number_input(
            "Ingrese las horas:",
            min_value=SLEPTIM1_data["opciones"]["min"],
            max_value=SLEPTIM1_data["opciones"]["max"],
            value=8, # Valor por defecto
            step=1,
            key="sleeptim1_q"
        )
        user_responses["SLEPTIM1"] = sleep_hours


        # Satifacción con la vida 

    st.subheader(LSATISFY_data["pregunta"])
    options_LSATISFY = [" "] + list(LSATISFY_data["opciones"].keys())
    selected_option_LSATISFY = st.selectbox(
        "Selecciona una opción:", # Etiqueta opcional para el selectbox
        options_LSATISFY,
        key="LSATISFY_q"
    )

    # Aquí es donde debes manejar la asignación a user_responses con una validación
    if selected_option_LSATISFY != " ":
        # Solo asigna si el usuario seleccionó una opción válida (no el placeholder)
        user_responses["LSATISFY"] = LSATISFY_data["opciones"][selected_option_LSATISFY]
    else:
        # Si el usuario no ha seleccionado nada o dejó el placeholder,
        # asigna None para indicar que no hay una respuesta válida
        user_responses["LSATISFY"] = None


        # Estado de salud

    st.subheader(PHYS14D_data["pregunta"])
    options_PHYS14D = [" "] + list(PHYS14D_data["opciones"].keys())
    selected_option_PHYS14D = st.selectbox(
        "Selecciona una opción:", # Etiqueta opcional para el selectbox
        options_PHYS14D,
        key="PHYS14D_q"
    )

    # Aquí es donde debes manejar la asignación a user_responses con una validación
    if selected_option_PHYS14D != " ":
        # Solo asigna si el usuario seleccionó una opción válida (no el placeholder)
        user_responses["PHYS14D"] = PHYS14D_data["opciones"][selected_option_PHYS14D]
    else:
        # Si el usuario no ha seleccionado nada o dejó el placeholder,
        # asigna None para indicar que no hay una respuesta válida
        user_responses["PHYS14D"] = None

    # --- 3. Limitaciones Funcionales y Historial Médico ---

        # Condición física, mental o emocional

    st.subheader(DECIDE_data["pregunta"])
    options_DECIDE = list(DECIDE_data["opciones"].keys())
    selected_option_DECIDE = st.radio(
        "Selecciona una opción:", # Etiqueta opcional para el selectbox
        options_DECIDE,
        key="DECIDE_q"
    )

    user_responses["DECIDE"] = DECIDE_data["opciones"][selected_option_DECIDE]


        # Recados

    st.subheader(DIFFALON_data["pregunta"])
    options_DIFFALON = list(DIFFALON_data["opciones"].keys())
    selected_option_DIFFALON = st.radio(
        "Selecciona una opción:", # Etiqueta opcional para el selectbox
        options_DIFFALON,
        key="DIFFALON_q"
    )

    user_responses["DIFFALON"] = DIFFALON_data["opciones"][selected_option_DIFFALON]


        # Diagnostico de enfermedades

    st.subheader(MICHD_data["pregunta"])
    options_MICHD = list(MICHD_data["opciones"].keys())
    selected_option_MICHD = st.radio(
        "Selecciona una opción:", # Etiqueta opcional para el selectbox
        options_MICHD,
        key="MICHD_q"
    )

    user_responses["MICHD"] = MICHD_data["opciones"][selected_option_MICHD]


    # --- 4. Acceso a Recursos y Comportamientos de Riesgo ---

        # Poder costear un medico

    st.subheader(MEDCOST1_data["pregunta"])
    options_MEDCOST1 = list(MEDCOST1_data["opciones"].keys())
    selected_option_MEDCOST1 = st.radio(
        "Selecciona una opción:", # Etiqueta opcional para el selectbox
        options_MEDCOST1,
        key="MEDCOST1_q"
    )

    user_responses["MEDCOST1"] = MEDCOST1_data["opciones"][selected_option_MEDCOST1]


        # Lista de enfermedades

    st.subheader(HIVRISK5_data["pregunta"])
    options_HIVRISK5 = list(HIVRISK5_data["opciones"].keys())
    selected_option_HIVRISK5 = st.radio(
        "Selecciona una opción:", # Etiqueta opcional para el selectbox
        options_HIVRISK5,
        key="HIVRISK5_q"
    )

    user_responses["HIVRISK5"] = HIVRISK5_data["opciones"][selected_option_HIVRISK5]

    # --- 5. Bienestar Emocional y Mental Directo ---

     # ¿Con qué frecuencia se siente socialmente aislado de los demás?

    st.subheader(SDHISOLT_data["pregunta"])
    options_SDHISOLT = [" "] + list(SDHISOLT_data["opciones"].keys())
    selected_option_SDHISOLT = st.selectbox(
        "Selecciona una opción:", # Etiqueta opcional para el selectbox
        options_SDHISOLT,
        key="SDHISOLT_q"
    )

    # Aquí es donde debes manejar la asignación a user_responses con una validación
    if selected_option_SDHISOLT != " ":
        # Solo asigna si el usuario seleccionó una opción válida (no el placeholder)
        user_responses["SDHISOLT"] = SDHISOLT_data["opciones"][selected_option_SDHISOLT]
    else:
        # Si el usuario no ha seleccionado nada o dejó el placeholder,
        # asigna None para indicar que no hay una respuesta válida
        user_responses["SDHISOLT"] = None


     # En los últimos 30 días, ¿cuántos días se ha sentido con un estado de salud no bueno (física o mentalmente)

    st.subheader(MENT14D_data["pregunta"])
    options_MENT14D = [" "] + list(MENT14D_data["opciones"].keys())
    selected_option_MENT14D = st.selectbox(
        "Selecciona una opción:", # Etiqueta opcional para el selectbox
        options_MENT14D,
        key="MENT14D_q"
    )

    # Aquí es donde debes manejar la asignación a user_responses con una validación
    if selected_option_MENT14D != " ":
        # Solo asigna si el usuario seleccionó una opción válida (no el placeholder)
        user_responses["MENT14D"] = MENT14D_data["opciones"][selected_option_MENT14D]
    else:
        # Si el usuario no ha seleccionado nada o dejó el placeholder,
        # asigna None para indicar que no hay una respuesta válida
        user_responses["MENT14D"] = None

    if st.button("Enviar Respuestas"):
        st.write("Respuestas recolectadas:", user_responses)
        # Aquí es donde pasarías user_responses al modelo para la predicción
        # prediction = your_model.predict(user_responses)
        # st.write(f"La predicción del modelo es: {prediction}")
        st.success("¡Respuestas enviadas! Ahora puedes procesarlas con tu modelo.")

# Llama a la función para ejecutar la encuesta
if __name__ == "__main__":
    run_survey()


# Prediccion    
if st.button("Predecir"):
    if None in respuestas_codificadas.values():
        st.warning("Por favor, responde todas las preguntas antes de continuar.")
    else:
        df_entrada = pd.DataFrame([respuestas_codificadas])
        pred = modelo.predict(df_entrada)
        st.success(f"✅ Predicción del modelo: {pred[0]}")
