from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS
import os



# Load your trained Random Forest model
model = joblib.load("rf_rank_predictor.pkl")

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["GET"])
def predict():
    try:
        # Extract features from query string
        overall_score = request.args.get("overall_score")
        industry_impact = request.args.get("industry_impact")
        international_outlook = request.args.get("international_outlook")

        # Validate inputs
        if overall_score is None or industry_impact is None or international_outlook is None:
            return jsonify({"error": "Missing input parameters"}), 400

        overall_score = float(overall_score)
        industry_impact = float(industry_impact)
        international_outlook = float(international_outlook)

        # Prepare input for prediction (must match training order)
        features = np.array([[overall_score, industry_impact, international_outlook]])
        pred = model.predict(features)[0]

        return jsonify({
            "predicted_rank": round(float(pred), 2)
        })
    except ValueError:
        return jsonify({"error": "Invalid input type"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
