from flask import Flask, render_template, request
import numpy as np
import pickle
import joblib
app = Flask(__name__)
filename = 'file_wine.pkl'
#model = pickle.load(open(filename, 'rb'))
model = joblib.load(filename)
#model = joblib.load(filename)
@app.route('/')
def index(): 
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    Alcohol = request.form['alcohol']
    Citric_Acid = request.form['citric_acid']
    Free_Sulfur_Dioxide = request.form['free_sulfur_dioxide']
    Sulphates = request.form['sulphates']
    PH = request.form['pH']

    
      
    pred = model.predict(np.array([[Alcohol, Citric_Acid, Free_Sulfur_Dioxide, Sulphates, PH ]]))
    print(pred)
    return render_template('index.html', predict=str(pred))


if __name__ == '__main__':
    app.run
