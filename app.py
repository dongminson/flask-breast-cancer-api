import os
from datetime import datetime
import json

import joblib
import numpy as np
from dotenv import load_dotenv
from flask import Flask, request
from sklearn.preprocessing import StandardScaler
from werkzeug.wrappers import Request, Response

app = Flask(__name__)

load_dotenv()

@app.route("/", methods=["GET"])
def predict():
    data = request.get_json()

    if "data" not in data:
        return "Error: 'data' key not found in JSON"

    new_data_point = []
    column_names = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
                    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
                    'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
                    'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
                    'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
                    'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
                    'concavity_worst', 'concave points_worst', 'symmetry_worst',
                    'fractal_dimension_worst']

    for column_name in column_names:
        if column_name not in data["data"]:
            return f"Error: '{column_name}' key not found in JSON data"

        value = data["data"][column_name]
        new_data_point.append(float(value))

    if len(new_data_point) != 30:
        return "Error: Invalid number of values. Expected 30 values."

    loaded_model = joblib.load('svm_rbf_model.joblib')
    sc = joblib.load('scaler.joblib')

    print(new_data_point)

    new_data_point = np.array(new_data_point)

    scaled_data_point = sc.transform(new_data_point.reshape(1, -1))

    prediction = loaded_model.predict(scaled_data_point)

    predicted_label = 'malignant' if prediction[0] == 1 else 'benign'

    response = {
        "prediction": predicted_label,
        "model": os.getenv('MODEL'),
        "version": os.getenv('VERSION'),
        "metadata": {
            "timestamp": datetime.now().isoformat()
        }
    }
    
    return json.dumps(response)
if __name__ == "__main__":
    from werkzeug.serving import run_simple
    run_simple('localhost', 9000, app)
    