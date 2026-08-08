# 🏭⚙️ Previsão de Entupimento em Válvulas Industriais usando Deep RNN

**Projeto Final da Disciplina de Introdução às Redes Neurais Profundas**

**Universidade Federal do Espírito Santo - UFES**

Este repositório contém o código e a análise de um modelo preditivo baseado em Deep Learning para identificar e classificar estágios de entupimento (degradação) em válvulas industriais. O modelo foi treinado utilizando dados reais de séries temporais cedidos por uma empresa.

# 📋 Resumo do Projeto

Na engenharia de manutenção preditiva, prever falhas antes que elas ocorram é fundamental para evitar paradas não programadas. Este projeto utiliza uma **Rede Neural Recorrente Profunda (Deep RNN)** com camadas **LSTM** para analisar o histórico recente de sensores de uma válvula e prever o seu estágio atual de operação.

## O Desafio dos Dados Industriais (O Paradoxo da Acurácia)

Durante o desenvolvimento, deparou-se com o clássico desafio de dados industriais: o **desbalanceamento extremo**. Como a válvula passa a maior parte da sua vida útil operando normalmente, um modelo padrão alcançava aproximadamente 95% de acurácia apenas "chutando" que o sistema estava sempre normal, ignorando completamente as falhas reais (Falsos Negativos).

## A Solução: Undersampling Assimétrico

Para resolver este problema de engenharia e encontrar o ponto de operação ideal (*Sweet Spot*), aplicou-se a técnica de **Undersampling Assimétrico**. Reduziu-se drasticamente a classe majoritária (operação normal) no conjunto de treinamento para equilibrar o aprendizado, mas mantevesse uma vantagem intencional de **+2%** em relação às anomalias. Isso evitou que a IA se tornasse "paranoica" (gerando muitos falsos alarmes), garantindo que ela aprendesse a física da falha de forma equilibrada.

# 🧠 Arquitetura e Metodologia

1. **Pré-processamento:**

    - Normalização dos dados contínuos (`Velocidade`, `Nível`, `Abertura`).
    - Transformação da variável categórica (`Distribuidor`).
    - Conversão das saídas binárias (`y1`, `y2`) em classes únicas: `Normal (00)`, `Leve (01)`, `Moderado (10)`, `Severo (11)`.

2. **Janela Deslizante (*Sliding Window*):**

    - Em vez de ler medições isoladas, o modelo recebe tensores 3D contendo blocos temporais (as últimas **10 medições**), permitindo que as camadas LSTM compreendam a tendência de degradação temporal do fluxo.

3. **Divisão Cronológica (*Data Splitting*):**

    - `70%` Treino | `15%` Validação | `15%` Teste.
    - A divisão foi estritamente cronológica, sem embaralhamento prévio, para evitar vazamento de dados do futuro para o passado (*Data Leakage*).

4. **Arquitetura da Deep RNN:**

    - **Input Layer:** (10 passos de tempo, 4 variáveis/*features*).
    - **Hidden Layer 1:** LSTM (64 neurônios) + Dropout (20%).
    - **Hidden Layer 2:** LSTM (32 neurônios) + Dropout (20%).
    - **Output Layer:** Dense (4 neurônios com ativação `softmax`).
    - **Otimizador:** Adam.

# 📊 Resultados Obtidos

Ao aplicar o balanceamento dinâmico assimétrico (+2%), o modelo apresentou uma evolução notável na capacidade de detecção de anomalias raras.

- **Redução de Falsos Positivos:** A pequena vantagem numérica para a classe "Normal" foi suficiente para frear alarmes falsos excessivos que ocorreriam em um modelo 50/50 perfeito.
- **Aumento do Recall (Sensibilidade)**: O modelo passou a identificar ativamente os estágios de entupimento que antes eram completamente ignorados pelo modelo desbalanceado.
- A análise visual através da **Curva de Aprendizado** e da **Matriz de Confusão** comprovam matematicamente a capacidade da rede de generalizar regras físicas para dados inéditos do mundo real.

# 💻 Como Executar

<big>**Pré-requisitos**</big>

Certifique-se de ter o Python instalado (recomendado >= 3.8) e instale as dependências necessárias:

```bash
   pip install pandas numpy tensorflow keras matplotlib seaborn scikit-learn
```

<big>**Rodando o script**</big>

1. Clone este repositório.
2. Certifique-se de que o arquivo `dados_limpos_e_normalizados.csv` esteja no mesmo diretório do script.
3. Execute o código:

```bash
   python Modelo_Deep_RNN_4.py
```

Ao final do treinamento (20 épocas), o script abrirá o gráfico da Curva de Aprendizado automaticamente. Feche a janela para visualizar a Matriz de Confusão.

# 🤝 Agradecimentos

Um agradecimento especial aos professores da disciplina de Introdução às Redes Neurais Profundas (UFES) pela orientação teórica, e à empresa parceira pela disponibilização dos dados reais de sensores que tornaram este estudo possível e aplicável ao cenário industrial.
