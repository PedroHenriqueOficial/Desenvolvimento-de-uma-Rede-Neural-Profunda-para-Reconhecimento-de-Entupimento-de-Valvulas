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
