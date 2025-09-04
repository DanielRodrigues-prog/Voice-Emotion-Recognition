
import librosa
import numpy as np
import pandas as pd
import os

# Mapa de emoções do dataset RAVDESS
emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Caminho para o diretório de dados
DATASET_PATH = r'C:\Users\rodridae\Desktop\Projeto SOM python\data'

def extract_features(file_path):
    """
    Extrai características de um arquivo de áudio.
    - MFCCs (Mel-Frequency Cepstral Coefficients)
    - Chroma (Pitch)
    - Mel Spectrogram
    """
    try:
        # Carrega o arquivo de áudio
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast') 
        
        # Extrai MFCCs
        # O resultado é uma matriz (n_mfcc, time_steps), então tiramos a média no eixo do tempo
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        
        # Extrai Chroma
        stft = np.abs(librosa.stft(audio))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        
        # Extrai Mel Spectrogram
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)
        
        # Concatena todas as características em um único vetor
        features = np.concatenate((mfccs, chroma, mel))
        
        return features
    except Exception as e:
        print(f"Erro ao processar {file_path}: {e}")
        return None

def create_features_dataframe():
    """
    Percorre o dataset, extrai características de cada arquivo de áudio
    e salva em um DataFrame do Pandas.
    """
    features_list = []
    
    # Percorre as pastas dos atores
    for actor_dir in os.listdir(DATASET_PATH):
        actor_path = os.path.join(DATASET_PATH, actor_dir)
        if os.path.isdir(actor_path):
            print(f"Processando: {actor_dir}")
            # Percorre os arquivos de áudio do ator
            for file_name in os.listdir(actor_path):
                if file_name.endswith('.wav'):
                    # Extrai a emoção do nome do arquivo
                    emotion_code = file_name.split('-')[2]
                    emotion = emotion_map.get(emotion_code)
                    
                    if emotion:
                        file_path = os.path.join(actor_path, file_name)
                        
                        # Extrai as características
                        data_features = extract_features(file_path)
                        
                        if data_features is not None:
                            features_list.append([data_features, emotion])

    # Cria o DataFrame
    df = pd.DataFrame(features_list, columns=['features', 'emotion'])
    return df

if __name__ == '__main__':
    print("Iniciando extração de características...")
    features_df = create_features_dataframe()
    
    # Divide a coluna 'features' em várias colunas para salvar no CSV
    # Cada linha em df['features'] é um array numpy, vamos expandi-lo
    expanded_features = pd.DataFrame(features_df['features'].values.tolist())
    final_df = pd.concat([expanded_features, features_df['emotion']], axis=1)

    # Salva o DataFrame em um arquivo CSV
    output_path = r'C:\Users\rodridae\Desktop\Projeto SOM python\data\audio_features.csv'
    final_df.to_csv(output_path, index=False)
    
    print(f"Extração concluída! Arquivo salvo em: {output_path}")
    print(f"Dimensões do DataFrame final: {final_df.shape}")