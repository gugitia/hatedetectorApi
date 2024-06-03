import onnxruntime as rt
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS

# Carregar o modelo ONNX /HateDetectorTreined/Train/HateDetector_Trained_Model.onnx
onnx_model_path = './HateDetectorTrained/Train/HateDetector_Trained_Model.onnx'
sess = rt.InferenceSession(onnx_model_path)

# Carrega o vetorizador 
with open('./HateDetectorTrained/Train/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_text = data['comment']
        
        if isinstance(vectorizer, TfidfVectorizer):
            X_input = vectorizer.transform([input_text])
        else:
            raise ValueError("O vetorizador não é uma instância de TfidfVectorizer.")
        
        X_input_np = X_input.toarray().astype('float32')

        # Obter os nomes das entradas e saídas do modelo ONNX
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

        # Fazer a previsão usando o modelo ONNX
        pred = sess.run([output_name], {input_name: X_input_np})

        # A predição é uma matriz numpy, você pode acessar o valor assim:
        prediction = float(pred[0][0][0])

        prediction_formatted = "{:.4f}".format(prediction)

        return jsonify({'prediction': prediction_formatted})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
        host = '127.0.0.1'
        port = 5000
        print(f'Server running at http://{host}:{port}')
        app.run(host='0.0.0.0', port=5000)
