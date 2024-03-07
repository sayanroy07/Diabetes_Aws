from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    Pregnancies = int(request.form.get('Pregnancies'))
    Glucose = int(request.form.get('Glucose'))
    Insulin = int(request.form.get('Insulin'))
    BMI = float(request.form.get('BMI'))
    DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
    Age = int(request.form.get('Age'))

    #Prediction
    result = model.predict(np.array([Pregnancies,Glucose,Insulin, BMI, DiabetesPedigreeFunction, Age]).reshape(1,6))

    return render_template('index.html',result=result)


@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)