# 🎙️ Voice Emotion Recognition

> Aplicação web que utiliza Machine Learning para detectar emoções humanas a partir de arquivos de áudio, analisando características acústicas da voz como MFCCs, energia RMS e ritmo — sem depender de transcrição de texto.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=flat&logo=flask&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## 📌 Sobre o projeto

Este projeto implementa um pipeline completo de reconhecimento de emoções em áudio:

1. **Extração de features acústicas** — MFCCs, Chroma, Energia-RMS, Tempo e mais, usando a biblioteca Librosa
2. **Treinamento do modelo** — `RandomForestClassifier` com hiperparâmetros otimizados via `GridSearchCV`
3. **API RESTful** — backend Flask que recebe arquivos de áudio e retorna a emoção prevista
4. **Interface web** — frontend HTML/CSS/JS para upload e visualização dos resultados

O modelo é capaz de classificar **8 emoções**: neutro, calmo, feliz, triste, com raiva, com medo, nojo e surpresa.

---

## 🚀 Funcionalidades

- Upload de arquivos `.wav` e `.mp3` pela interface web
- Análise 100% acústica, sem dependência de transcrição de texto
- Modelo otimizado com GridSearchCV
- API RESTful pronta para integração com outros sistemas
- Estrutura modular: extração de features e treinamento separados da API

---

## 🛠️ Tecnologias

| Camada | Tecnologias |
|---|---|
| Backend | Python 3.8+, Flask |
| Machine Learning | Scikit-learn, Librosa, Pandas, NumPy |
| Frontend | HTML5, CSS3, JavaScript |

---

## ⚙️ Como executar localmente

### Pré-requisitos

- Python 3.8+
- pip

### 1. Clone o repositório

```bash
git clone https://github.com/DanielRodrigues-prog/Voice-Emotion-Recognition.git
cd Voice-Emotion-Recognition
```

### 2. Crie e ative um ambiente virtual

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Configure as variáveis de ambiente

```bash
cp .env.example .env
# Edite o .env se necessário
```

### 5. (Opcional) Treine o modelo do zero

O modelo `.pkl` já está incluído no repositório. Para replicar o processo de treinamento:

```bash
# Baixe o dataset RAVDESS: https://zenodo.org/record/1188976
# Descompacte em data/

# Extraia as features
python src/features_extractor.py

# Treine o modelo
python src/train_model.py
```

### 6. Inicie a aplicação

```bash
python app.py
```

Acesse `http://127.0.0.1:5000` no navegador.

---

## 📁 Estrutura do projeto

```
Voice-Emotion-Recognition/
├── models/                  # Modelos treinados (.pkl)
├── notebook/                # Jupyter Notebooks de experimentação
├── src/
│   ├── features_extractor.py  # Extração de features acústicas
│   └── train_model.py         # Treinamento e avaliação do modelo
├── static/                  # Assets do frontend (CSS, JS)
├── templates/               # Templates HTML (Flask/Jinja2)
├── app.py                   # Servidor Flask e endpoints da API
├── test_api.py              # Testes da API
├── requirements.txt         # Dependências Python
├── .env.example             # Exemplo de variáveis de ambiente
└── .gitignore
```

---

## 🔌 API

### `POST /predict`

Recebe um arquivo de áudio e retorna a emoção detectada.

**Request:**
```
Content-Type: multipart/form-data
Body: audio (file) — .wav ou .mp3
```

**Response:**
```json
{
  "emotion": "happy",
  "confidence": 0.87
}
```

---

## 📊 Dataset

Este projeto utiliza o dataset [RAVDESS](https://zenodo.org/record/1188976) (Ryerson Audio-Visual Database of Emotional Speech and Song), com 24 atores gravando em 8 emoções diferentes.

O dataset não está incluído no repositório devido ao tamanho. Faça o download no link acima e descompacte em `data/`.

---

## 🧪 Testes

```bash
python test_api.py
```

---

## 📄 Licença

Este projeto está licenciado sob a [MIT License](LICENSE).

---

## 👤 Autor

**Daniel Rodrigues**
- GitHub: [@DanielRodrigues-prog](https://github.com/DanielRodrigues-prog)
- LinkedIn: [daniel-rodrigues-10305b239](https://www.linkedin.com/in/daniel-rodrigues-10305b239/)
