import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV # <-- Adicionado GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import joblib
import os
import matplotlib.pyplot as plt

# 1. Carregar os dados
print("Carregando o dataset de características...")
df = pd.read_csv(r'C:\Users\rodridae\Desktop\Projeto SOM python\Projeto-Som-Python\data\audio_features.csv')

# 2. Preparar os dados para o treinamento
X = df.drop('emotion', axis=1)
y = df['emotion']

# 3. Dividir os dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Normalização dos dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Definir o "grid" de parâmetros para testar
param_grid = {
    'n_estimators': [200, 300],          # Número de árvores
    'max_depth': [15, 25, None],         # Profundidade máxima
    'min_samples_leaf': [1, 2],          # Mínimo de amostras por folha
    'criterion': ['gini', 'entropy']     # Critério de divisão
}

# 6. Criar o modelo base e o objeto GridSearchCV
# n_jobs=-1 usa todos os núcleos do seu processador para acelerar o processo
# cv=3 faz uma validação cruzada com 3 "folds" (dobras)
# verbose=2 mostra o progresso do treinamento no terminal
print("\nIniciando a otimização de hiperparâmetros com GridSearchCV...")
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# 7. Treinar o GridSearchCV
grid_search.fit(X_train_scaled, y_train)

# 8. Pegar o melhor modelo encontrado pelo GridSearch
print("\nOtimização concluída!")
print("Melhores parâmetros encontrados:")
print(grid_search.best_params_)
best_model = grid_search.best_estimator_

# 9. Avaliar o modelo encontrado
y_pred = best_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAcurácia do modelo no conjunto de teste: {accuracy * 100:.2f}%")

# 10. Gerar a Matriz de Confusão para o MELHOR modelo
print("\nGerando a Matriz de Confusão para o melhor modelo...")
labels = sorted(y_test.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation='vertical')
plt.title('Matriz de Confusão (Modelo Otimizado)')
output_path = '../confusion_matrix_optimized.png'
plt.savefig(output_path)
print(f"Matriz de Confusão otimizada salva em: {output_path}")

# 11. Salvar o MELHOR modelo e o scaler
model_path = '../models/emotion_model_optimized.pkl' 
scaler_path = '../models/scaler.pkl' # 
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(best_model, model_path)
joblib.dump(scaler, scaler_path)
print(f"\nMelhor modelo salvo em: {model_path}")
print(f"Scaler salvo em: {scaler_path}")
