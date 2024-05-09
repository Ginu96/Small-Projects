
# %%
import os
import numpy as np
from flask import Flask, render_template,request
import pickle
import joblib


#%%
app = Flask(__name__)
model = pickle.load(open('dt.pkl', 'rb'))
# Load the scaling parameters
scaling_params = joblib.load('scaling_params.pkl')

#web-app page

@app.route('/')
def home():
    return render_template('Index.html')



# Function to perform scaling on input data
def scale_data(data):
    scaled_data = {}
    for feature in scaling_params:
        mean = scaling_params[feature]['mean']
        std = scaling_params[feature]['std']
        scaled_feature = (data[feature] - mean) / std
        scaled_data[feature] = scaled_feature
    return scaled_data

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {}
        for i in scaling_params:
            input_data[i] = float(request.form[i])

        # Scale the variables
        scaled_input = scale_data(input_data)

        # Get the values of the other four variables
        other_variables = [float(request.form[var]) for var in ["DataPlan", "DataUsage", "CustServCalls", "OverageFee"]]

        # Combine the scaled variables and the other four variables
        input_features = list(scaled_input.values()) + other_variables

        # Convert the input features into a NumPy array
        final_features = np.array([input_features])

        # Perform the prediction
        prediction = model.predict(final_features)

        if prediction[0] == 1:
            result = "Oops! You lost this customer"
        else:
            result = "Happy! This customer stays with you"

        return render_template('Index.html', prediction_text=result)
    except Exception as e:
        return f'Error: {str(e)}'


# %%
# Run the Flask server
if(__name__=='__main__'):
    app.run(debug=True)  
    
#host="127.0.0.1", port=8000


# %%
