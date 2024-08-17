from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np


def data_preprocesing(data_input):
    # Transform user value input for marital status for ML model
    if data_input[0] == "married":
        marital_status = 1
    else:
        marital_status = 0
    # Transform user value input for education level for ML model
    if data_input[1] == "master" or data_input[1] == "phd":
        education_level = 1
    else:
        education_level = 0
    # Transform user value for credit score for ML model
    if data_input[2] == "670-739" or data_input[2] == "740-799" or data_input[2] == "800-850":
        credit_score = 1
    else:
        credit_score = 0

    # Transform user value for applicant income for ML model
    if data_input[3] == "0-25000":
        applicant_income = 25000
    elif data_input[3] == "25001-50000":
        applicant_income = 50000
    elif data_input[3] == "50001-75000":
        applicant_income = 75000
    elif data_input == "75001-100000":
        applicant_income = 100000
    elif data_input == "100001-150000":
        applicant_income = 150000
    else:
        applicant_income = 175000
    # Transform user value for co-applicant income for ML model
    if data_input[4] == "0-25000":
        coapp_income = 25000
    elif data_input[4] == "25001-50000":
        coapp_income = 50000
    elif data_input[4] == "50001-75000":
        coapp_income = 75000
    elif data_input == "75001-100000":
        coapp_income = 100000
    elif data_input == "100001-150000":
        coapp_income = 150000
    else:
        coapp_income = 175000

    total_income = applicant_income + coapp_income
    total_income_log = np.log(total_income)
    data = [marital_status, education_level, credit_score, total_income_log]
    columnas = ['Married', 'Education', 'Credit_History', 'Total_Income_log']
    df = pd.DataFrame(columns=columnas)
    df.loc[0] = data
    return df


app = Flask(__name__)

# Load the ML model


def load_model(model_path):
    model = joblib.load(model_path)
    return model


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    sl = request.form['sl']
    sw = request.form['sw']
    pl = request.form['pl']
    pw = request.form['pw']
    pz = request.form['pz']

    model_path = 'le_model.pickle'
    model = load_model(model_path)
    # Process the input data and make predictions using your model
    # Here you need to transform the input data as expected by your model
    # Adjust according to your model's input format
    input_data = [sl, sw, pl, pw, pz]
    input_data_ml = data_preprocesing(input_data)
    print(input_data_ml.head())
    prediction = model.predict(input_data_ml)  # Example prediction

    result = 'Eligible' if prediction[0] == 1 else 'Not Eligible'
    return jsonify({'result': result})


if __name__ == '__main__':
    app.run(debug=True)
