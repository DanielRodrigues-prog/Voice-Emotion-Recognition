
import requests
import os

# URL do endpoint da sua API
API_URL = 'http://127.0.0.1:5000/predict'

# Caminho para um arquivo de áudio que você quer testar

TEST_AUDIO_PATH = os.path.join()

def test_prediction():
    print(f"Enviando o arquivo: {TEST_AUDIO_PATH}")
    
    try:
        # Abre o arquivo de áudio em modo de leitura binária ('rb')
        with open(TEST_AUDIO_PATH, 'rb') as audio_file:
            file_payload = {'file': audio_file}
            
            # Envia a requisição POST para a API
            response = requests.post(API_URL, files=file_payload)
            
            # Verifica o status da resposta
            if response.status_code == 200:
                print("\nPrevisão recebida com sucesso!")
                print("="*30)
                print(f"Emoção Prevista: {response.json().get('emotion')}")
                print("="*30)
            else:
                print(f"\nErro ao fazer a previsão. Status: {response.status_code}")
                print(f"Resposta do servidor: {response.text}")

    except requests.exceptions.ConnectionError as e:
        print("\nErro de conexão. Você se lembrou de iniciar o servidor Flask (app.py)?")
        print(f"Detalhe do erro: {e}")
    except FileNotFoundError:
        print(f"\nErro: O arquivo de áudio não foi encontrado em '{TEST_AUDIO_PATH}'")
    except Exception as e:
        print(f"\nUm erro inesperado ocorreu: {e}")

if __name__ == '__main__':
    test_prediction()
