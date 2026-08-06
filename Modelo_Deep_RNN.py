# Importação de bibliotecas

import os
import random
import pandas as pd
import numpy as np

# Fixando a semente da aleatoriedade para garantir reprodutibilidade

os.environ['PYTHONHASHSEED'] = '42'
np.random.seed(42)
random.seed(42)

# Importação das bibliotecas de gráficos e métricas

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Importação do Keras/TensorFlow

import tensorflow as tf
tf.random.set_seed(42)
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input

""" --- CARREGAMENTO E PREPARAÇÃO DOS DADOS --- """

print("Carregando os dados limpos ...")
df = pd.read_csv('dados_limpos_e_normalizados.csv')

"""
• 0 representa o estágio 00 (Operação normal - sem entupimento); 
• 1 representa o estágio 01 (Entupimento leve); 
• 2 representa o estágio 10 (Entupimento moderado); 
• 3 representa o estágio 11 (Obstrução severa - totalmente entupido). 
"""

# Para facilitar para a rede neural, transformamos y1 e y2, que são binários, em uma única classe variando de 0 a 3.
# Matemática binária: (y1 * 2^1) + (y2 * 2^0) = y1*2 + y2

df['Estagio'] = df['y1'] * 2 + df['y2']
recursos_X = df[['Velocidade', 'Nivel', 'Abertura', 'DI']].values
alvo_y = df['Estagio'].values

""" --- CRIAÇÃO DA JANELA DESLIZANTE (SLIDING WINDOW) --- """

# Definindo quantos "passos de tempo" (linhas do passado) a RNN vai pegar.
# A rede vai olhar as últimas 10 medições para prever a próxima.

TAMANHO_JANELA = 10 

def criar_sequencias_temporais(X, y, tamanho_janela):

    # Transforma dados tabulares 2D em matrizes 3D para a Deep RNN.

    X_seq, y_seq = [], []

    for i in range(len(X) - tamanho_janela):
        X_seq.append(X[i:(i + tamanho_janela)]) # Pega 10 linhas do passado.
        y_seq.append(y[i + tamanho_janela]) # Pega o estágio de entupimento logo após essas 10 linhas.

    return np.array(X_seq), np.array(y_seq)

print("\nCriando sequências temporais com janela deslizante ...")
X_3D, y_1D = criar_sequencias_temporais(recursos_X, alvo_y, TAMANHO_JANELA)

""" --- DIVISÃO EM TREINO, VALIDAÇÃO E TESTE (ORDEM CRONOLÓGICA) --- """

print("\nDivisão dos dados de forma cronologica ...")

"""
• Os primeiros 70% das linhas para treino; 
• Os 15% seguintes para validação; 
• E os últimos 15% para testes. 
"""

tamanho_total = len(X_3D)

corte_treino = int(tamanho_total * 0.70)
corte_validacao = int(tamanho_total * 0.85)

X_treino = X_3D[:corte_treino]
y_treino = y_1D[:corte_treino]

X_validacao = X_3D[corte_treino:corte_validacao]
y_validacao = y_1D[corte_treino:corte_validacao]

X_teste = X_3D[corte_validacao:]
y_teste = y_1D[corte_validacao:]

""" --- BALANCEAMENTO DINÂMICO ASSIMÉTRICO DOS DADOS --- """

print("\nAplicando o undersampling assimétrico nos dados de treinamento ...")

# Descobre qual é a classe que mais aparece

valores_unicos, contagens = np.unique(y_treino, return_counts=True)
classe_majoritaria = valores_unicos[np.argmax(contagens)]

indices_majoritarios = np.where(y_treino == classe_majoritaria)[0]
indices_minoritarios = np.where(y_treino != classe_majoritaria)[0]

qtd_minoritarias = len(indices_minoritarios)

print(f"\n\t- Classe Maioritária Encontrada (Estágio {classe_majoritaria}): {len(indices_majoritarios)} amostras")
print(f"\t- Classes Minoritárias (Anomalias): {qtd_minoritarias} amostras")

if qtd_minoritarias > 0:

    # A classe maioritária teve, aproximadamente, 1000 amostras a mais (2%) se comparado com a classe minoritária.
    # Em vez de pegar 1 para 1, pegamos 2% a mais da classe maioritária.
    # Ex: Se tiver 1000 minoritárias, vai pegar 1020 maioritárias.
    
    multiplicador_maioritaria = 1.02
    qtd_majoritaria_nova = int(qtd_minoritarias * multiplicador_maioritaria)
    
    # Prevenção de segurança: garante que não vamos tentar pegar mais dados do que existem.

    qtd_majoritaria_nova = min(qtd_majoritaria_nova, len(indices_majoritarios))
    
    # Sorteamos a nova quantidade

    indices_major_reduzidos = np.random.choice(indices_majoritarios, size=qtd_majoritaria_nova, replace=False)
    
    # Juntamos os índices

    indices_balanceados = np.concatenate([indices_major_reduzidos, indices_minoritarios])
    np.random.shuffle(indices_balanceados)
    
    # Atualização do treino

    X_treino = X_treino[indices_balanceados]
    y_treino = y_treino[indices_balanceados]
    
    print(f"\nNovo treino:\n\tAnomalias: {qtd_minoritarias}\n\tNormais: {qtd_majoritaria_nova}.")
    print(f"\nTotal do treino balanceado: {len(X_treino)} amostras.")

else:
    print("\nNenhuma anomalia encontrada no conjunto de treino. Logo, os dados não foram balanceados.")

""" --- CONSTRUÇÃO DA ARQUITETURA DA DEEP RNN --- """

print("\nConstruindo o modelo Deep RNN ...\n")

# Foi utilizado o modelo sequencial

modelo = Sequential()
modelo.add(Input(shape=(TAMANHO_JANELA, X_treino.shape[2])))

# Parâmetros da primeira camada oculta

modelo.add(LSTM(64, return_sequences=True))
modelo.add(Dropout(0.2))

# Parâmetros da segunda camada oculta

modelo.add(LSTM(32, return_sequences=False))
modelo.add(Dropout(0.2))

# Parâmetros da camada de saída

modelo.add(Dense(4, activation='softmax'))

# Definindo como a rede aprende

modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Exibe as características finais da Deep RNN

modelo.summary() 

""" --- TREINAMENTO DO MODELO --- """

print("\nIniciando o treino com dados assimetricamente balanceados ...\n")

# batch_size=256: A rede vai processar 256 janelas temporais de uma vez antes de atualizar os pesos.
# epochs=20: O algoritmo vai ler todos os dados de treino 20 vezes.

historico = modelo.fit(X_treino, y_treino, epochs=20, batch_size=256, validation_data=(X_validacao, y_validacao), verbose=1)

""" --- VALIDAÇÃO FINAL --- """

print("\nValidação com dados de teste no mundo real desbalanceado ...")
perda, acuracia = modelo.evaluate(X_teste, y_teste, verbose=0)
print(f"\nAcurácia final no teste: {acuracia * 100:.2f}%\n")

""" --- CURVA DE APRENDIZADO E MATRIZ DE CONFUSÃO (GRÁFICOS) --- """

# Gráfico da Curva de Aprendizado (Loss).

plt.figure(figsize=(10, 5))
plt.plot(historico.history['loss'], label='Erro no Treino (Loss)', color='blue', linewidth=2)
plt.plot(historico.history['val_loss'], label='Erro na Validação (Val Loss)', color='orange', linewidth=2)
plt.title('Curva de Aprendizagem (Undersampling Assimétrico)', fontsize=14)
plt.xlabel('Épocas (Epochs)')
plt.ylabel('Erro (Loss)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show() 

# Gráfico da Matriz de Confusão.

previsoes_probabilidades = modelo.predict(X_teste)
previsoes_classes = np.argmax(previsoes_probabilidades, axis=1)

classes_presentes_matriz = np.unique(np.concatenate([y_teste, previsoes_classes]))
dicionario_estagios = {0: 'Normal (00)', 1: 'Leve (01)', 2: 'Moderado (10)', 3: 'Severo (11)'}
nomes_eixos = [dicionario_estagios[c] for c in classes_presentes_matriz]

matriz = confusion_matrix(y_teste, previsoes_classes, labels=classes_presentes_matriz)

plt.figure(figsize=(8, 6))
sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', xticklabels=nomes_eixos, yticklabels=nomes_eixos)
plt.title('Matriz de Confusão - Undersampling Assimétrico (+2%)', fontsize=14)
plt.ylabel('Estágio Real (Gabarito)')
plt.xlabel('Estágio Previsto pela Rede')
plt.show()