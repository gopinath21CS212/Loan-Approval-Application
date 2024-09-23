import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# Load the trained model from file
model = pickle.load(open("LoanFinal.pickle", 'rb'))

# Home route to render the HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route to handle POST requests and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    # For rendering results on HTML GUI
    input_data = [float(x) for x in request.form.values()]
    final_features = [np.array(input_data)]
    prediction = model.predict(final_features)
    output = np.around(prediction)

    if output == 1:
        output = "Yes"
    elif output == 0:
        output = "No"

    return render_template('index.html', prediction_text='Loan Approval Status: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
