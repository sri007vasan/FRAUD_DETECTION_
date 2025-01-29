from flask import Flask, render_template, request, jsonify
import pandas as pd
from model import predict_fraud, get_fraud_chain

app = Flask(__name__)

# Load dataset
df = pd.read_csv("transactions.csv")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    data = request.json
    person_id = int(data["person_id"])
    
    # Predict fraud likelihood
    fraud_percentage = predict_fraud(person_id)
    
    # If fraud likelihood > 70%, get the fraud chain
    if fraud_percentage > 70:
        fraud_chain = get_fraud_chain(person_id)
        return jsonify({
            "fraud_percentage": fraud_percentage,
            "fraud_chain": fraud_chain
        })
    else:
        return jsonify({
            "fraud_percentage": fraud_percentage,
            "fraud_chain": None
        })

if __name__ == "__main__":
    app.run(debug=True)
