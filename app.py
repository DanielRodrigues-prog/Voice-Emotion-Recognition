
import os
import joblib
import librosa
import numpy as np
from flask import Flask, request, jsonify

# Inicializa a aplicação Flask
app = Flask(__name__)

# Carrega o modelo e o scaler UMA VEZ quando a aplicação inicia.
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

# Esta é a MESMA função que usamos no script de extração.
def extract_features(file):
    try:
        # 'file' aqui pode ser um caminho ou um objeto de arquivo aberto
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

# --- Rota da API ---
@app.route('/predict', methods=['POST'])
def predict():
    # Verifica se os modelos foram carregados
    if not model or not scaler:
        return jsonify({'error': 'Modelo ou scaler não foram carregados.'}), 500

    # Verifica se um arquivo foi enviado na requisição
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo de áudio foi enviado.'}), 400

    file = request.files['file']

    # Verifica se o nome do arquivo está vazio
    if file.filename == '':
        return jsonify({'error': 'Nome de arquivo vazio.'}), 400

    if file:
        # 1. Extrai as características do áudio enviado
        features = extract_features(file)
        if features is None:
            return jsonify({'error': 'Não foi possível processar o arquivo de áudio.'}), 500
        
        # 2. Prepara as características para o modelo
        # O modelo espera um array 2D, então remodelamos (reshape)
        features_reshaped = features.reshape(1, -1)
        
        # 3. Aplica o MESMO scaler que usamos no treinamento
        features_scaled = scaler.transform(features_reshaped)
        
        # 4. Faz a previsão
        prediction = model.predict(features_scaled)
        
        # 5. Retorna o resultado como JSON
        emotion = prediction[0]
        return jsonify({'emotion': emotion})

    return jsonify({'error': 'Erro inesperado.'}), 500

# --- Execução da Aplicação ---
if __name__ == '__main__':
    # 'debug=True' faz com que o servidor reinicie automaticamente após alterações no código.
    # Não use em produção!
    app.run(debug=True)