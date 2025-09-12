import os
import joblib
import librosa
print(f"--- VERSÃO DO LIBROSA SENDO USADA PELA API: {librosa.__version__} ---")

import numpy as np
import pandas as pd  # Adicionado para corrigir o UserWarning
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# --- Carregamento dos Modelos ---
# Carregando o modelo OTIMIZADO e o scaler correspondente
models_dir = 'models'
model_path = os.path.join(models_dir, 'emotion_model_optimized.pkl') # <-- Caminho atualizado
scaler_path = os.path.join(models_dir, 'scaler.pkl')

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("* Modelo OTIMIZADO e Scaler carregados com sucesso!")
except Exception as e:
    print(f"* Erro ao carregar os modelos: {e}")
    model = None
    scaler = None

# --- FUNÇÃO DE EXTRAÇÃO DE CARACTERÍSTICAS ATUALIZADA ---
# Esta é a versão final, com todas as 182 características.
def extract_features(file):
    try:
        audio, sample_rate = librosa.load(file, res_type='kaiser_fast') 
        
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        stft = np.abs(librosa.stft(audio))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)
        rms = np.mean(librosa.feature.rms(y=audio).T, axis=0)
        # Usando o caminho corrigido para a função tempo
        tempo = librosa.feature.tempo(y=audio, sr=sample_rate)[0]

        # Concatena todas as características
        features = np.concatenate((mfccs, chroma, mel, rms, [tempo]))
        return features
    except Exception as e:
        print(f"Erro na extração de características: {e}")
        return None

# --- Rota da Página Inicial (sem mudanças) ---
@app.route('/')
def home():
    return render_template('index.html')

# --- Rota da API (com a correção do UserWarning) ---
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
        features = extract_features(file) # <-- Agora extrai 182 features
        if features is None:
            return jsonify({'error': 'Não foi possível processar o arquivo de áudio.'}), 500
        
        features_reshaped = features.reshape(1, -1)
        
        # Correção para o UserWarning
        feature_names = [str(i) for i in range(features_reshaped.shape[1])]
        features_df = pd.DataFrame(features_reshaped, columns=feature_names)
        
        features_scaled = scaler.transform(features_df) # <-- Agora recebe 182 e espera 182
        
        prediction = model.predict(features_scaled)
        emotion = prediction[0]
        return jsonify({'emotion': emotion})
    return jsonify({'error': 'Erro inesperado.'}), 500

# --- Execução da Aplicação (sem mudanças) ---
if __name__ == '__main__':
    app.run(debug=True)