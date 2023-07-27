import numpy as np
from flask import Flask, request,render_template
import pickle

#app = Flask(__name__)
app = Flask(__name__, template_folder='templates') 
model = pickle.load(open('SVM.pickle', 'rb'))

@app.route('/')
def home():
    return render_template('HomePage.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input from form and print it
    input_features = [float(x) for x in request.form.values()]
    print('Received input:', input_features)
    # Convert input to numpy array and print final input
    final_features = [np.array(input_features)]
    print('Final input:', final_features)
    # Make prediction using the model
    prediction = model.predict(final_features)

   
    return render_template('HomePage.html', prediction_text='Predicted Iris Class: {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)