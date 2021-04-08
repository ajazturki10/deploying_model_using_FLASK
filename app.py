import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
model = joblib.load(open("wine_quality_predictor.pkl", "rb"))

@app.route('/')
def home():
    return render_template('wine_index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():

    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    if output == 1.0:
        return render_template('wine_index.html', pred = 'Good')
    else:
        return render_template('wine_index.html', pred = 'Harmful')

@app.route('/results',methods=['POST', 'GET'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)