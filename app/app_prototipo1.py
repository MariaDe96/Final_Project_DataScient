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

preguntas_dict = {
    "SDHISOLT": {
        "pregunta": "¿Con qué frecuencia se siente socialmente aislado de los demás?",
        "opciones": {
            "Siempre": 1,
            "Usualmente": 1,
            "Algunas veces": 1,
            "Rara vez": 0,
            "Nunca": 0
        }
    },
    "_MENT14D": {
        "pregunta": "En los ultimos 30 días, cuantos dias se ha sentido con un estado de salud no bueno.",
        "opciones": {
            "0 dias": 0,
            "1 a 13 días": 1,
            "Más de 13 días": 2
        }
    },
    "LSATISFY": {
        "pregunta": "En general, ¿está satisfecho con su vida?",
        "opciones": {
            "Muy satisfecho": 0,
            "Satisfecho": 1,
            "Insatisfecho": 2,
            "Muy insatisfecho": 3
        }
    },
    "_PHYS14D": {
        "pregunta": "Estado de salud física no bueno: 0 días, 1-13 días, 14-30 días",
        "opciones": {
            "0 dias": 0,
            "1 a 13 días": 1,
            "Más de 13 días": 2
        }
    },
    "SLEPTIM1": {
        "pregunta": "De media, ¿cuántas horas duermes en un periodo de 24 horas?",
        "opciones": {
            "type": "numerical_range",
            "min": 1,
            "max": 24
        }
    },
    "_INCOMG1": {
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
    },
    "_BMI5CAT": {
        "pregunta": "A cual categorías de índice de masa corporal pertenece?",
        "opciones": {
            "Bajo peso": 0,
            "Peso normal": 1,
            "Sobre peso": 2,
            "Obesidad": 3
        }
    },
    "_AGEG5YR": {
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
    },
    "_EDUCAG": {
        "pregunta": "Nivel de estudios",
        "opciones": {
            "No se graduó de la escuela secundaria": 0,
            "Se graduó de la escuela secundaria": 1,
            "Asistió a la Universidad o Escuela Técnica": 2,
            "Graduado de Colegio o Escuela Técnica": 3
        }
    },
    "DECIDE": {
        "pregunta": "Debido a una condición física, mental o emocional, ¿tiene usted serias dificultades para concentrarse, recordar o tomar decisiones?",
        "opciones": {
            "No": 0,
            "Sí": 1
        }
    },
    "SEXVAR": {
        "pregunta": "Sexo del encuestado",
        "opciones": {
            "Masculino": 0,
            "Mujer": 1
        }
    },
    "HHADULT": {
        "pregunta": "¿Cuántos miembros de su hogar, incluido usted mismo, tienen 18 años de edad o más?",
        "opciones": {
            "type": "numerical_range",
            "min": 1,
            "max": 20
        }
    },
    "DIFFALON": {
        "pregunta": "Debido a una condición física, mental o emocional, ¿tiene usted dificultad para hacer recados solo, como visitar al consultorio del médico o ir de compras?",
        "opciones": {
            "No": 0,
            "Sí": 1
        }
    },
    "_MICHD": {
        "pregunta": "Encuestados que alguna vez informaron haber tenido una enfermedad coronaria (EC) o un infarto de miocardio (IM)",
        "opciones": {
            "No": 0,
            "Sí": 1
        }
    },
    "_CHLDCNT": {
        "pregunta": "Número de niños en el hogar",
        "opciones": {
            "No hay": 0,
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "Hay 5 o más": 5
        }
    },
    "MEDCOST1": {
        "pregunta": "¿Hubo alguna ocasión en los últimos 12 meses en la que usted necesitó ir al médico pero no pudo porque no podía costearlo?",
        "opciones": {
            "No": 0,
            "Sí": 1
        }
    },
    "HIVRISK5": {
        "pregunta": "Voy a leerle una lista. Cuando termine, por favor dígame si alguna de las situaciones se aplica a usted. No necesita decirme cuál. Se ha inyectado alguna droga distinta a las recetadas para usted en el último año. Ha recibido tratamiento para una enfermedad de transmisión sexual o ETS en el último año. Ha dado o recibido dinero o drogas a cambio de sexo en el último año.",
        "opciones": {
            "No": 0,
            "Sí": 1
        }
    },
    "MARITAL": {
        "pregunta": "Estado civil",
        "opciones": {
            "Casado": 0,
            "Divorciado": 1,
            "Viudo": 2,
            "Separado": 3,
            "Nunca me casé": 4,
            "Un miembro de una pareja no casada": 5
        }
    },
    "RENTHOM1": {
        "pregunta": "¿Es usted propietario o alquila su vivienda?",
        "opciones": {
            "Propio": 0,
            "Alquilado": 1,
            "Otro arreglo": 2
        }
    },
    "EMPLOY1": {
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
}

# Diccionario para guardar las respuestas codificadas
respuestas_codificadas = {}

# Recorrer las preguntas dinámicamente
for columna, contenido in preguntas_dict.items():
    opciones = ["Selecciona una opción..."] + list(contenido["opciones"].keys())
    respuesta = st.selectbox(contenido["pregunta"], opciones, key=columna)

    if respuesta != "Selecciona una opción...":
        respuestas_codificadas[columna] = contenido["opciones"][respuesta]
    else:
        respuestas_codificadas[columna] = None

st.write(respuestas_codificadas)
# # Mostramos solo las claves como opciones
# opcion = st.selectbox("¿Con qué frecuencia se siente socialmente aislado de los demás? (Ultimo)", list(MENT14D_dict.keys()))
# st.write("Has elegido:", opcion)

# # Codificamos la respuesta
# valor_codificado = MENT14D_dict[opcion]
# st.write("Valor codificado:", valor_codificado)

# respuesta = st.radio("¿Con qué frecuencia se siente socialmente aislado de los demás?", list(MENT14D_dict.keys()))

if st.button("Predecir"):
    if None in respuestas_codificadas.values():
        st.warning("Por favor, responde todas las preguntas antes de continuar.")
    else:
        df_entrada = pd.DataFrame([respuestas_codificadas])
        pred = modelo.predict(df_entrada)
        st.success(f"✅ Predicción del modelo: {pred[0]}")
