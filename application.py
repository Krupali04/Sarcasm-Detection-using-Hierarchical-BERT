from flask import Flask,request,render_template 
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData( 
            country= request.form.get('country'),
            year= int(request.form.get('year')),
            status= request.form.get('status'),
            adult_mortality=float(request.form.get('adult_mortality')),
            infant_deaths= int(request.form.get('infant_deaths')),
            alcohol= float(request.form.get('alcohol')),
            percentage_expenditure= float(request.form.get('percentage_expenditure')),
            hepatitis_b= float(request.form.get('hepatitis_b')),
            measles= int(request.form.get('measles')),
            under_five_deaths= int(request.form.get('under_five_deaths')),
            polio= float(request.form.get('polio')),
            total_expenditure= float(request.form.get('total_expenditure')),
            diphtheria= float(request.form.get('diphtheria')),
            hiv_aids= float(request.form.get('hiv_aids')),
            population= int(request.form.get('population')),
            thinness_1_19_years= float(request.form.get('thinness_1_19_years')),
            thinness_5_9_years= float(request.form.get('thinness_5_9_years')),
            income_composition= float(request.form.get('income_composition')),
            bmi= float(request.form.get('bmi')),
            gdp= float(request.form.get('gdp')),
            schooling= float(request.form.get('schooling'))
        )

        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000)        