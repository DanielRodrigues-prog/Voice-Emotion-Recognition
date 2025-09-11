import os
import joblib
import librosa
import numpy as np
# MODIFICADO: Adicione render_template
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# --- Carregamento dos Modelos (nenhuma mudança aqui) ---
models_dir = 'models'
model_path = os.path.join(models_dir, 'emotion_model.pkl')
scaler_path = os.path.join(models_dir, 'scaler.pkl')

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("* Modelo e Scaler carregados com sucesso!")
except Exception as e:
    print(f"* Erro ao carregar os modelos: {e}")
    model = None
    scaler = None

# --- Função de Extração de Características (nenhuma mudança aqui) ---
def extract_features(file):
    try:
        audio, sample_rate = librosa.load(file, res_type='kaiser_fast') 
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        stft = np.abs(librosa.stft(audio))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)
        features = np.concatenate((mfccs, chroma, mel))
        return features
    except Exception as e:
        print(f"Erro na extração de características: {e}")
        return None

# --- NOVA ROTA PARA A PÁGINA INICIAL ---
@app.route('/')
def home():
    # Renderiza e retorna o arquivo index.html da pasta 'templates'
    return render_template('index.html')

# --- Rota da API (nenhuma mudança aqui) ---
@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        return jsonify({'error': 'Modelo ou scaler não foram carregados.'}), 500
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo de áudio foi enviado.'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nome de arquivo vazio.'}), 400
    if file:
        features = extract_features(file)
        if features is None:
            return jsonify({'error': 'Não foi possível processar o arquivo de áudio.'}), 500
        features_reshaped = features.reshape(1, -1)
        features_scaled = scaler.transform(features_reshaped)
        prediction = model.predict(features_scaled)
        emotion = prediction[0]
        return jsonify({'emotion': emotion})
    return jsonify({'error': 'Erro inesperado.'}), 500

# --- Execução da Aplicação (nenhuma mudança aqui) ---
if __name__ == '__main__':
    app.run(debug=True)