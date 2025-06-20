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

# Interfaz
st.title("Predicción con modelo ML")


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

_MENT14D_data = {
    "pregunta": "En los ultimos 30 días, cuantos dias se ha sentido con un estado de salud no bueno.",
    "opciones": {
        "0 dias": 0,
        "1 a 13 días": 1,
        "Más de 13 días": 2
    }
}

LSATISFY_data = {
    "pregunta": "En general, ¿está satisfecho con su vida?",
    "opciones": {
        "Muy satisfecho": 0,
        "Satisfecho": 1,
        "Insatisfecho": 2,
        "Muy insatisfecho": 3
    }
}

_PHYS14D_data = {
    "pregunta": "Estado de salud física no bueno: 0 días, 1-13 días, 14-30 días",
    "opciones": {
        "0 dias": 0,
        "1 a 13 días": 1,
        "Más de 13 días": 2
    }
}

SLEPTIM1_data = {
    "pregunta": "De media, ¿cuántas horas duermes en un periodo de 24 horas?",
    "opciones": {
        "type": "numerical_range",
        "min": 1,
        "max": 24
    }
}

INCOMG1_data = {
    "pregunta": "En cual de las siguientes categorías de ingresos considera que esta.",
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
    "pregunta": "A cual categorías de índice de masa corporal pertenece?",
    "opciones": {
        "Bajo peso": 0,
        "Peso normal": 1,
        "Sobre peso": 2,
        "Obesidad": 3
    }
}

AGEG5YR_data = {
    "pregunta": "A cual de las siguientes categoría de edad pertence?",
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

EDUCAG_data = {
    "pregunta": "Nivel de estudios",
    "opciones": {
        "No se graduó de la escuela secundaria": 0,
        "Se graduó de la escuela secundaria": 1,
        "Asistió a la Universidad o Escuela Técnica": 2,
        "Graduado de Colegio o Escuela Técnica": 3
    }
}

DECIDE_data = {
    "pregunta": "Debido a una condición física, mental o emocional, ¿tiene usted serias dificultades para concentrarse, recordar o tomar decisiones?",
    "opciones": {
        "No": 0,
        "Sí": 1
    }
}

SEXVAR_data = {
    "pregunta": "Sexo del encuestado",
    "opciones": {
        "Masculino": 0,
        "Mujer": 1
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

DIFFALON_data = {
    "pregunta": "Debido a una condición física, mental o emocional, ¿tiene usted dificultad para hacer recados solo, como visitar al consultorio del médico o ir de compras?",
    "opciones": {
        "No": 0,
        "Sí": 1
    }
}

MICHD_data = {
    "pregunta": "Encuestados que alguna vez informaron haber tenido una enfermedad coronaria (EC) o un infarto de miocardio (IM)",
    "opciones": {
        "No": 0,
        "Sí": 1
    }
}

CHLDCNT_data = {
    "pregunta": "Número de niños en el hogar",
    "opciones": {
        "No hay": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "Hay 5 o más": 5
    }
}

MEDCOST1_data = {
    "pregunta": "¿Hubo alguna ocasión en los últimos 12 meses en la que usted necesitó ir al médico pero no pudo porque no podía costearlo?",
    "opciones": {
        "No": 0,
        "Sí": 1
    }
}

HIVRISK5_data = {
    "pregunta": "Voy a leerle una lista. Cuando termine, por favor dígame si alguna de las situaciones se aplica a usted. No necesita decirme cuál. Se ha inyectado alguna droga distinta a las recetadas para usted en el último año. Ha recibido tratamiento para una enfermedad de transmisión sexual o ETS en el último año. Ha dado o recibido dinero o drogas a cambio de sexo en el último año.",
    "opciones": {
        "No": 0,
        "Sí": 1
    }
}

MARITAL_data = {
    "pregunta": "Estado civil",
    "opciones": {
        "Casado": 0,
        "Divorciado": 1,
        "Viudo": 2,
        "Separado": 3,
        "Nunca me casé": 4,
        "Un miembro de una pareja no casada": 5
    }
}

RENTHOM1_data = {
    "pregunta": "¿Es usted propietario o alquila su vivienda?",
    "opciones": {
        "Propio": 0,
        "Alquilado": 1,
        "Otro arreglo": 2
    }
}

EMPLOY1_data = {
    "pregunta": "¿Actualmente usted…?",
    "opciones": {
        "Empleado por salario": 0,
        "Trabajadores por cuenta propia": 1,
        "Sin trabajo durante 1 año o más": 2,
        "Sin trabajo por menos de 1 año": 3,
        "Una ama de casa": 4,
        "Un estudiante": 5,
        "Jubilado": 6,
        "Incapaz de trabajar": 7
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
    "_MENT14D": _MENT14D_data,
    "LSATISFY": LSATISFY_data,
    "_PHYS14D": _PHYS14D_data,
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

    # Ejemplo para una pregunta con opciones de selección (radio o selectbox)
    st.subheader(SDHISOLT_data["pregunta"])
    selected_option_sdhisolt = st.radio(
        "Seleccione una opción:",
        list(SDHISOLT_data["opciones"].keys()),
        key="sdhisolt_q"
    )
    user_responses["SDHISOLT"] = SDHISOLT_data["opciones"][selected_option_sdhisolt]

    # Ejemplo para una pregunta con rango numérico (number_input)
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

    # Repite este patrón para cada una de tus preguntas, adaptando el widget de Streamlit
    # según el tipo de "opciones" (diccionario para selectbox/radio, o numerical_range para number_input).

    # Puedes agrupar los diccionarios individuales en una lista o un diccionario para iterar
    # si prefieres un enfoque más programático en lugar de llamar a cada uno manualmente.
    # Por ejemplo:
    all_question_data = {
        "SDHISOLT": SDHISOLT_data,
        "_MENT14D": _MENT14D_data,
        "LSATISFY": LSATISFY_data,
        # ... y así sucesivamente con todos los diccionarios individuales
    }

    st.subheader("Otras preguntas (ejemplo iterativo):")
    for key, data in all_question_data.items():
        if key not in user_responses: # Para no duplicar las preguntas ya hechas arriba
            if "type" in data["opciones"] and data["opciones"]["type"] == "numerical_range":
                st.write(data["pregunta"])
                value = st.number_input(
                    f"Ingrese un valor para {key}:",
                    min_value=data["opciones"]["min"],
                    max_value=data["opciones"]["max"],
                    key=f"input_{key}"
                )
                user_responses[key] = value
            else:
                st.write(data["pregunta"])
                selected_option = st.radio(
                    "Seleccione una opción:",
                    list(data["opciones"].keys()),
                    key=f"radio_{key}"
                )
                user_responses[key] = data["opciones"][selected_option]


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
