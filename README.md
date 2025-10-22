# Detector de Emoções em Áudio

Este projeto é uma aplicação web completa que utiliza Machine Learning para analisar um arquivo de áudio e detectar a emoção expressa na voz do interlocutor. A análise é feita inteiramente com base nas características acústicas da voz, sem depender da transcrição do texto.

## Funcionalidades

- **Interface Web:** Permite o upload de arquivos de áudio (`.wav`, `.mp3`) para análise.
- **Análise por IA:** Extração de múltiplas características acústicas (MFCCs, Energia-RMS, Tempo, etc.).
- **Modelo Otimizado:** Um modelo `RandomForestClassifier` com hiperparâmetros ajustados via `GridSearchCV` prevê uma das 8 emoções.
- **API RESTful:** O modelo é servido através de uma API construída com Flask, pronta para ser consumida.

## Tecnologias Utilizadas

- **Back-end:** Python, Flask
- **Machine Learning:** Scikit-learn, Librosa, Pandas, Numpy
- **Front-end:** HTML, CSS, JavaScript

## Como Executar

1.  **Clone o Repositório**
    ```bash
    git clone https://github.com/DanielRodrigues-prog/Projeto-Som-Python.git
    cd Projeto-Som-Python
    ```

2.  **Baixe o Dataset**
    - Faça o download do dataset [RAVDESS](https://zenodo.org/record/1188976) (Audio_Speech_Actors_01-24.zip).
    - Descompacte e coloque o conteúdo dentro da pasta `data/`. (Este passo é necessário porque o dataset é muito grande para ser incluído no repositório).

3.  **Instale as Dependências**
    ```bash
    python -m pip install -r requirements.txt
    ```

4.  **Execute os Scripts de Preparação**
    - O modelo (`.pkl`) já está incluído no repositório, então você não precisa treinar para rodar a aplicação. Contudo, se quiser replicar o processo de treinamento:
    ```bash
    # (Opcional) Gere o arquivo de características:
    python src/features_extractor.py
    # (Opcional) Treine o modelo:
    python src/train_model.py
    ```

5.  **Inicie a Aplicação**
    ```bash
    python app.py
    ```
6.  Abra seu navegador e acesse `http://127.0.0.1:5000`.

## Autor

* **Daniel Rodrigues**
* **LinkedIn:** (https://www.linkedin.com/in/daniel-rodrigues-10305b239/)
