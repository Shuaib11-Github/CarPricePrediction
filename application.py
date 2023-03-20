from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

## Route the app
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods = ['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            km_driven = request.form.get('km_driven'),
            fuel = request.form.get('fuel'),
            seller_type = request.form.get('seller_type'),
            transmission = request.form.get('transmission'),
            owner = request.form.get('owner'),
            seats = request.form.get('seats'),
            torque_rpm = request.form.get('torque_rpm'),
            mil_kmpl = request.form.get('mil_kmpl'),
            engine_cc = request.form.get('engine_cc'),
            max_power_new = request.form.get('max_power_new'),
            No_of_years = request.form.get('No_of_years')
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        return render_template('home.html', results = results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0", debug = True)