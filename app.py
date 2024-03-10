from flask import Flask, render_template, request, jsonify, url_for
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict' , methods = ['POST'])
def predict():
    Age = float(request.form.get('Age'))
    Hypertension = int(request.form.get('Hypertension'))
    Heartdisease = int(request.form.get('Heartdisease'))
    BMI = float(request.form.get('BMI'))
    HbA1c_level = float(request.form.get('HbA1c_level'))
    blood_glucose_level = int(request.form.get('blood_glucose_level'))
    gender_Male = int(request.form.get('gender_Male'))
    gender_Other = int(request.form.get('gender_Other'))
    smoker_past_smoker = int(request.form.get('smoker_past_smoker'))
    smoker_smoker = int(request.form.get('smoker_smoker'))
    #Prediction
    result = model.predict(np.array([Age,Hypertension,Heartdisease,BMI,HbA1c_level,blood_glucose_level,gender_Male,gender_Other,smoker_past_smoker,smoker_smoker]).reshape(1,10))
    if int(result) == 0:
        result = "You don't have Diabetes, have some Sweets"
    else:
        result = "You are Diabetic"
    return render_template('index.html' , result=result)

@app.route('/predict_api',methods=['POST'])
def predict_api():
    # For Direct API Calls
    data = request.get_json(force=True)
    result = model.predict(np.array([Pregnancies,Glucose,Insulin, BMI, DiabetesPedigreeFunction, Age]).reshape(1,6))
    output = result[0]
    return jsonify(object)


if __name__ == '__main__':
    app.run(debug=True)