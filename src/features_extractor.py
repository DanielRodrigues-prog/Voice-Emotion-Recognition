import librosa
import numpy as np
import pandas as pd
import os

# Mapa de emoções 
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

#ATENÇÃO: Verifique se este caminho está correto para sua estrutura de pastas ---
DATASET_PATH = r''


def extract_features(file_path):
    """
    Extrai características de um arquivo de áudio.
    - MFCCs (Mel-Frequency Cepstral Coefficients)
    - Chroma (Pitch)
    - Mel Spectrogram
    - RMS (Energy)
    - Tempo (Pace) <-- NOVA CARACTERÍSTICA
    """
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast') 
        
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        stft = np.abs(librosa.stft(audio))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)
        rms = np.mean(librosa.feature.rms(y=audio).T, axis=0)
        
        # Extrai a estimativa de Tempo (BPM)
        # librosa.beat.tempo retorna um array, então pegamos o primeiro elemento [0]
        tempo = librosa.beat.tempo(y=audio, sr=sample_rate)[0]
        # -------------------------

        # Concatena TODAS as características. Precisamos colocar 'tempo' em uma lista para concatenar.
        features = np.concatenate((mfccs, chroma, mel, rms, [tempo]))
        # ------------------
        
        return features
    except Exception as e:
        print(f"Erro ao processar {file_path}: {e}")
        return None

def create_features_dataframe():
    features_list = []
    
    for actor_dir in os.listdir(DATASET_PATH):
        actor_path = os.path.join(DATASET_PATH, actor_dir)
        if os.path.isdir(actor_path):
            print(f"Processando: {actor_dir}")
            for file_name in os.listdir(actor_path):
                if file_name.endswith('.wav'):
                    emotion_code = file_name.split('-')[2]
                    emotion = emotion_map.get(emotion_code)
                    if emotion:
                        file_path = os.path.join(actor_path, file_name)
                        data_features = extract_features(file_path)
                        if data_features is not None:
                            features_list.append([data_features, emotion])
    
    df = pd.DataFrame(features_list, columns=['features', 'emotion'])
    return df

if __name__ == '__main__':
    print("Iniciando extração de características (versão 3.0 com Tempo)...")
    features_df = create_features_dataframe()
    
    expanded_features = pd.DataFrame(features_df['features'].values.tolist())
    final_df = pd.concat([expanded_features, features_df['emotion']], axis=1)
    
    # --- ATENÇÃO: Verifique se este caminho está correto para sua estrutura de pastas ---
    output_path = r''
    final_df.to_csv(output_path, index=False)
    
    print(f"Extração concluída! Arquivo salvo em: {output_path}")
    print(f"Dimensões do DataFrame final: {final_df.shape}")
