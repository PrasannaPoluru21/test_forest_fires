#creating web application with flask
import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application=Flask(__name__)
app=application

#import ridge regressor and standard scaler pickle
with open('models/scaler.pkl', 'rb') as f:
    standard_scaler = pickle.load(f)

with open('models/ridge.pkl', 'rb') as fq:
    ridge_model = pickle.load(fq)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        Temperature= float(request.form.get('Temperature'))
        RH= float(request.form.get('RH'))
        Ws= float(request.form.get('Ws'))
        Rain= float(request.form.get('Rain'))
        FFMC= float(request.form.get('FFMC'))
        DMC= float(request.form.get('DMC'))
        ISI= float(request.form.get('ISI'))
        Classes= float(request.form.get('Classes'))
        Region= float(request.form.get('Region'))

        print(f"Received input: {Temperature}, {RH}, {Ws}, {Rain}, {FFMC}, {DMC}, {ISI}, {Classes}, {Region}")
        print(type(Temperature))

        new_data = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        new_data_scaled = standard_scaler.transform(new_data)       
        result= ridge_model.predict(new_data_scaled)

        return render_template('home.html', results=result[0])

    else:
        return render_template('home.html')

if __name__=='__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)