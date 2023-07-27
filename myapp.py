import logging
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = pickle.load(open('loan_model.pkl', 'rb'))

# Configure logging
logging.basicConfig(level=logging.INFO)


@app.route('/')
def hello():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.content_type == 'application/json':
            # If the content type is JSON, access the data using request.json
            data = request.json
            print("Received JSON data:", data)

            # Get the input values from the data dictionary and convert to integers
            ApplicantIncome = int(data.get('ApplicantIncome', 0))
            CoapplicantIncome = int(data.get('CoapplicantIncome', 0))
            LoanAmount = int(data.get('LoanAmount', 0))
            Credit_History = int(data.get('Credit_History', 0))
        else:
            # If the content type is form data, access the data using request.form
            data = request.form
            print("Received form data:", data)

            # Get the input values from the form data
            ApplicantIncome = int(request.form['ApplicantIncome'])
            CoapplicantIncome = int(request.form['CoapplicantIncome'])
            LoanAmount = int(request.form['LoanAmount'])
            Credit_History = int(request.form['Credit_History'])

        # Create an input array for prediction
        final_features = np.array([[ApplicantIncome, CoapplicantIncome, LoanAmount, Credit_History]])

        print("Final Features:", final_features)

        # Make prediction using the model
        prediction = model.predict(final_features)

        print("Prediction:", prediction)

        # Assuming that the model predicts 1 for eligibility and 0 for non-eligibility
        status = 'Congratulations! You are eligible for a loan. 😀' if prediction == 'Y' else 'We are sorry, you are not eligible for a loan at the moment.'

        return jsonify({'prediction': status})

    except Exception as e:
        # Log the exception and return an error response
        logging.error(str(e))
        return jsonify({'error': 'An error occurred during prediction.'})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
