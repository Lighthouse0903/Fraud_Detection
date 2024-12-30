from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        type_transaction = request.form["type"]
        step = int(request.form["step"])
        amount = float(request.form["amount"])
        oldbalanceOrg = float(request.form["oldbalanceOrg"])
        newbalanceOrig = float(request.form["newbalanceOrig"])
        oldbalanceDest = float(request.form["oldbalanceDest"])
        newbalanceDest = float(request.form["newbalanceDest"])

        type_map = {"PAYMENT": 0, "TRANSFER": 1, "CASH_OUT": 2, "DEBIT": 3, "CASH_IN": 4}
        type_value = type_map.get(type_transaction, -1)

        data = {
            "step": step,
            "type": type_value,
            "amount": amount,
            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig,
            "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest
        }
        features = pd.DataFrame(data, index=[0])

        prediction = model.predict(features)
        prediction_prob = model.predict_proba(features)

        result = {
            "is_fraud": bool(prediction[0]),
            "probability_no_fraud": round(prediction_prob[0][0] * 100, 2),
            "probability_fraud": round(prediction_prob[0][1] * 100, 2),
        }

        return render_template(
            "index.html",
            result=result,
            type=type_transaction
        )
    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=True)
