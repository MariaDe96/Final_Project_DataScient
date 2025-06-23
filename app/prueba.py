from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)
with open("modelo_catboost.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        input_values = {k: request.form.get(k) for k in request.form if k != "imc"}
        imc = float(request.form.get("imc", 0))
        if imc < 18.5:
            input_values['_BMI5CAT'] = 0
        elif imc < 24.9:
            input_values['_BMI5CAT'] = 1
        elif imc < 29.9:
            input_values['_BMI5CAT'] = 2
        else:
            input_values['_BMI5CAT'] = 3
        input_df = pd.DataFrame([input_values])
        prediction = model.predict(input_df)
        resultado = "Posible depresión" if prediction[0] == 1 else "No presenta depresión"
        return render_template("index.html", resultado=resultado)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
