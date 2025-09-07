from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__, template_folder="../frontend/templates")

model = pickle.load(open("model/sales_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            tv = float(request.form['TV'])
            radio = float(request.form['Radio'])
            newspaper = float(request.form['Newspaper'])

            features = np.array([[tv, radio, newspaper]])
            features = scaler.transform(features)

            prediction = round(model.predict(features)[0], 2)
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
