# src/train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib # Usado para salvar o modelo

# 1. Carregar os dados
print("Carregando o dataset de características...")
df = pd.read_csv(r'C:\Users\rodridae\Desktop\Projeto SOM python\data\audio_features.csv')

# Verifica as primeiras linhas e informações do dataframe
print("Primeiras 5 linhas do dataset:")
print(df.head())
print("\nInformações do DataFrame:")
df.info()

# 2. Preparar os dados para o treinamento
# X contém as características (features), y contém os rótulos (labels)
X = df.drop('emotion', axis=1)
y = df['emotion']

# 3. Dividir os dados em conjuntos de treinamento e teste
# 80% para treino, 20% para teste
# random_state garante que a divisão seja a mesma toda vez que rodarmos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nFormato dos dados de treino: {X_train.shape}")
print(f"Formato dos dados de teste: {X_test.shape}")

# 4. Normalização dos dados (Feature Scaling)
# É crucial para que características com valores maiores não dominem o modelo.
# Usamos o StandardScaler para isso.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # Importante: usamos o mesmo scaler treinado nos dados de treino

# 5. Treinar o modelo de Machine Learning
# Usaremos o RandomForestClassifier, que é um modelo robusto e bom para começar.
print("\nTreinando o modelo RandomForestClassifier...")
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)
print("Treinamento concluído!")

# 6. Avaliar o modelo
# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test_scaled)

# Calcular a acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAcurácia do modelo no conjunto de teste: {accuracy * 100:.2f}%")

# 7. Salvar o modelo e o scaler
# Salvar o modelo treinado e o scaler para que possamos usá-los depois na nossa API.
model_path = '../models/emotion_model.pkl'
scaler_path = '../models/scaler.pkl'

# Criar a pasta 'models' se ela não existir
import os
os.makedirs('../models', exist_ok=True)

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print(f"\nModelo salvo em: {model_path}")
print(f"Scaler salvo em: {scaler_path}")