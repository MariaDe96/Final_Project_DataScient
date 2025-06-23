from flask import Flask, request, render_template_string
import pickle
import numpy as np

app = Flask(__name__)

# Cargamos el modelo
with open("modelo_catboost.pkl", "rb") as f:
    model = pickle.load(f)

# Diccionarios de opciones en español -> valor numérico
opciones = {
    "SEXVAR": {"Hombre": 0, "Mujer": 1},
    "_AGEG5YR": {"45-49 años": 5, "50-54 años": 6, "55-59 años": 7, "60-64 años": 8, "65-69 años": 9, "70-74 años": 10, "75-79 años": 11, "80 o más": 12},
    "MARITAL": {"Casado": 0, "Divorciado": 1, "Viudo": 2, "Separado": 3, "Soltero": 4, "Pareja no casada": 5},
    "_EDUCAG": {"Sin secundaria": 0, "Graduado de secundaria": 1, "Universidad o técnica (no graduado)": 2, "Universidad o técnica (graduado)": 3},
    "EMPLOY1": {"Empleado": 0, "Autónomo": 1, "Desempleado >1 año": 2, "Desempleado <1 año": 3, "Amas de casa": 4, "Estudiante": 5, "Jubilado": 6, "Incapaz de trabajar": 7},
    "_INCOMG1": {"Menos de $15,000": 0, "$15,000 a $25,000": 1, "$25,000 a $35,000": 2, "$35,000 a $50,000": 3, "$50,000 a $100,000": 4, "$100,000 a $200,000": 5, "Más de $200,000": 6},
    "_ASTHMS1": {"Actual": 1, "Anterior": 2, "Nunca": 3},
    "_AIDTST4": {"No": 0, "Sí": 1},
    "DECIDE": {"No": 0, "Sí": 1},
    "SDHISOLT": {"Siempre/A veces": 1, "Rara vez/Nunca": 0},
    "LSATISFY": {"Muy satisfecho": 0, "Satisfecho": 1, "Insatisfecho": 2, "Muy insatisfecho": 3},
    "_PHYS14D": {"Cero días": 0, "1-13 días": 1, "14+ días": 2},
    "_MENT14D": {"Cero días": 0, "1-13 días": 1, "14+ días": 2},
    "_DRDXAR2": {"No": 0, "Sí": 1},
    "_BMI5CAT": {"Bajo peso": 0, "Peso normal": 1, "Sobrepeso": 2, "Obeso": 3},
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        data = []
        for key in ["SEXVAR","_AGEG5YR","MARITAL","_EDUCAG","EMPLOY1","_INCOMG1","_ASTHMS1","_AIDTST4","_DRDXAR2","_BMI5CAT","_PHYS14D","SLEPTIM1","DECIDE","SDHISOLT","LSATISFY","_MENT14D"]:
            value = request.form.get(key)
            if key == "SLEPTIM1":
                data.append(int(value))
            else:
                data.append(opciones.get(key, {}).get(value, value))
        prediction = model.predict(np.array(data).reshape(1, -1))[0]
    
    return render_template_string("""
<html>
<body>
<h1>Predicción de Depresión</h1>
<form method="POST">
<h2>Preguntas generales o demográficas</h2>
<select name="SEXVAR">{% for k in opciones.SEXVAR.keys() %}<option>{{k}}</option>{% endfor %}</select><br>
<select name="_AGEG5YR">{% for k in opciones._AGEG5YR.keys() %}<option>{{k}}</option>{% endfor %}</select><br>
<select name="MARITAL">{% for k in opciones.MARITAL.keys() %}<option>{{k}}</option>{% endfor %}</select><br>
<select name="_EDUCAG">{% for k in opciones._EDUCAG.keys() %}<option>{{k}}</option>{% endfor %}</select><br>
<select name="EMPLOY1">{% for k in opciones.EMPLOY1.keys() %}<option>{{k}}</option>{% endfor %}</select><br>
<select name="_INCOMG1">{% for k in opciones._INCOMG1.keys() %}<option>{{k}}</option>{% endfor %}</select><br>
<h2>Preguntas sobre salud física o enfermedades</h2>
<select name="_ASTHMS1">{% for k in opciones._ASTHMS1.keys() %}<option>{{k}}</option>{% endfor %}</select><br>
<select name="_AIDTST4">{% for k in opciones._AIDTST4.keys() %}<option>{{k}}</option>{% endfor %}</select><br>
<select name="_DRDXAR2">{% for k in opciones._DRDXAR2.keys() %}<option>{{k}}</option>{% endfor %}</select><br>
<select name="_BMI5CAT">{% for k in opciones._BMI5CAT.keys() %}<option>{{k}}</option>{% endfor %}</select><br>
<select name="_PHYS14D">{% for k in opciones._PHYS14D.keys() %}<option>{{k}}</option>{% endfor %}</select><br>
<h2>Preguntas sobre hábitos de sueño</h2>
<input type="number" name="SLEPTIM1" min="1" max="24" required><br>
<h2>Preguntas sobre salud mental</h2>
<select name="DECIDE">{% for k in opciones.DECIDE.keys() %}<option>{{k}}</option>{% endfor %}</select><br>
<select name="SDHISOLT">{% for k in opciones.SDHISOLT.keys() %}<option>{{k}}</option>{% endfor %}</select><br>
<select name="LSATISFY">{% for k in opciones.LSATISFY.keys() %}<option>{{k}}</option>{% endfor %}</select><br>
<select name="_MENT14D">{% for k in opciones._MENT14D.keys() %}<option>{{k}}</option>{% endfor %}</select><br>
<button type="submit">Predecir</button>
</form>
{% if prediction is not none %}<h3>Resultado de predicción: {{'Depresión' if prediction == 1 else 'No depresión'}}</h3>{% endif %}
</body>
</html>
""", opciones=opciones, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
