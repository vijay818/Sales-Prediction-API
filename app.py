import pandas as pd
import numpy as np
import xgboost as xgb
from flask import Flask, request, jsonify, render_template
import pickle
from flask_debugtoolbar import DebugToolbarExtension

app = Flask(__name__)
#app.url_map.strict_slashes = False
#app.debug = True
app.config['SECRET_KEY'] = 'DontTellAnyone'

toolbar = DebugToolbarExtension(app)

model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/result',methods=['POST'])
def result():
    '''
    For rendering results on HTML GUI
    
    '''
    columns = ['Store', 'CompetitionDistance', 'Promo', 'Promo2', 'SchoolHoliday',
       'StoreType', 'Assortment', 'StateHoliday', 'DayOfWeek', 'Month', 'Day',
       'Year', 'WeekOfYear', 'CompetitionOpen', 'PromoOpen', 'IsPromoMonth']
    int_features = [int(x) for x in request.form.values()]
    df = pd.DataFrame()
    s = pd.Series(int_features)
    df = df.append([s], ignore_index = True)
    df.columns = columns
    final_features = xgb.DMatrix(df)
    prediction = model.predict(final_features)
    output = int(prediction[0])
    return render_template('form.html', prediction_text = 'Sales prediction: {}'.format(output))
    #return 'Hello World'


if __name__ == "__main__":
    app.run(debug=True)