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
input_valor = st.number_input("Introduce un valor:", value=0.0)
input_valor = st.number_input("Que tan feliz eres:", value=0.0)
input_valor = st.number_input("ejemplo 2:", value=0.0)
input_valor = st.number_input("ejemplo 3:", value=0.0)
input_valor = st.number_input("ejemplo 4:", value=0.0)
input_valor = st.number_input("ejemplo 5:", value=0.0)
input_valor = st.number_input("ejemplo 6:", value=0.0)
input_valor = st.number_input("ejemplo 7:", value=0.0)
input_valor = st.number_input("ejemplo 8:", value=0.0)
input_valor = st.number_input("ejemplo 9:", value=0.0)
input_valor = st.number_input("ejemplo 10:", value=0.0)
input_valor = st.number_input("ejemplo 11:", value=0.0)
input_valor = st.number_input("ejemplo 12:", value=0.0)
input_valor = st.number_input("ejemplo 13:", value=0.0)
input_valor = st.number_input("ejemplo 14:", value=0.0)
input_valor = st.number_input("ejemplo 15:", value=0.0)
input_valor = st.number_input("ejemplo 16:", value=0.0)
input_valor = st.number_input("ejemplo 17:", value=0.0)
input_valor = st.number_input("ejemplo 18:", value=0.0)
input_valor = st.number_input("ejemplo 19:", value=0.0)

if st.button("Predecir"):
    # pred = modelo.predict(np.array([[input_valor]]))
    st.success(f"Resultado: {pred[0]}")