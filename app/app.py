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

if st.button("Predecir"):
    # pred = modelo.predict(np.array([[input_valor]]))
    st.success(f"Resultado: {pred[0]}")