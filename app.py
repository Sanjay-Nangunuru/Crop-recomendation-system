from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)  # Enable CORS

model = joblib.load('crop_recommendation_model.pkl')

@app.route('/')
def home():
    return "<h2>Crop Recommendation API is running</h2><p>Use the <code>/predict</code> endpoint with POST requests to get predictions.</p>"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        features = [data[key] for key in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
        prediction = model.predict([features])
        return jsonify({'recommended_crop': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
