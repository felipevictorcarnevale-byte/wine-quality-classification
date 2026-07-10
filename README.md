# Classificação da Qualidade de Vinhos com Machine Learning

Projeto de classificação binária que prevê se um vinho é de **alta qualidade** (nota ≥ 7) ou de **baixa/média qualidade** (nota < 7) a partir de suas características físico-químicas, utilizando a base *Wine Quality* (vinho tinto).

> Tech Challenge — Fase 2 (POSTECH)

---

## Contexto e objetivo

A avaliação tradicional da qualidade de um vinho é feita por especialistas, por meio de análises sensoriais — um processo subjetivo, demorado e dependente da experiência do avaliador. Este projeto investiga se é possível **prever a qualidade do vinho a partir de medições físico-químicas obtidas durante a produção**, oferecendo uma ferramenta de apoio à decisão para enólogos e produtores.

O problema foi tratado como uma **classificação binária**:

- **Alta qualidade (1):** nota ≥ 7
- **Baixa/Média qualidade (0):** nota < 7

---

## Base de dados

- **Fonte:** [Wine Quality Dataset (WineQT)](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset) — Kaggle
- **Amostras:** 1.143 vinhos tintos
- **Variáveis:** 11 características físico-químicas + nota de qualidade

| Variável | Descrição |
|---|---|
| fixed acidity | Acidez fixa |
| volatile acidity | Acidez volátil |
| citric acid | Ácido cítrico |
| residual sugar | Açúcar residual |
| chlorides | Cloretos |
| free sulfur dioxide | Dióxido de enxofre livre |
| total sulfur dioxide | Dióxido de enxofre total |
| density | Densidade |
| pH | pH |
| sulphates | Sulfatos |
| alcohol | Teor alcoólico |
| quality | Nota de qualidade (variável de origem do alvo) |

O carregamento é feito automaticamente via `kagglehub`, não sendo necessário baixar o arquivo manualmente.

---

## Principais decisões e achados

**Análise exploratória (EDA)**
- Base sem valores faltantes; 125 duplicatas identificadas (desconsiderando o identificador `Id`) e removidas no pré-processamento.
- **Classes fortemente desbalanceadas:** apenas **13,9%** das amostras são de alta qualidade. Esse desbalanceamento é o principal desafio do problema e guiou toda a estratégia de modelagem e avaliação.
- As variáveis mais correlacionadas com a qualidade foram **teor alcoólico (+0,48)**, **acidez volátil (−0,41)** e **sulfatos (+0,26)**.
- Outliers identificados pela regra do IQR foram mantidos, por representarem características plausíveis de vinhos reais e pela robustez dos modelos baseados em árvore.

**Pré-processamento**
- Remoção de duplicatas e descarte das colunas `Id` e `quality` na modelagem.
- Divisão treino/teste **estratificada** (80/20), preservando a proporção das classes.
- Padronização com `StandardScaler` (ajustada apenas no treino, evitando vazamento de dados).
- **Feature engineering:** criação de `sulfur_ratio` (enxofre livre / total) e `total_acidity` (acidez fixa + volátil).
- **SMOTE** aplicado **somente no conjunto de treino** para tratar o desbalanceamento.

---

## Modelos e resultados

Foram treinados e comparados **seis modelos**, combinando diferentes algoritmos e estratégias de balanceamento (`class_weight` e SMOTE). A avaliação priorizou **F1-score e recall da classe "Alta Qualidade"**, métricas mais adequadas que a acurácia em cenários desbalanceados.

| # | Modelo | Recall | Precision | F1 | ROC-AUC |
|---|---|---|---|---|---|
| 1 | **XGBoost + SMOTE** | **0,778** | 0,568 | **0,656** | 0,896 |
| 2 | Random Forest + SMOTE | 0,667 | 0,600 | 0,632 | 0,909 |
| 3 | XGBoost | 0,593 | 0,615 | 0,604 | 0,897 |
| 4 | Regressão Logística | 0,741 | 0,364 | 0,488 | 0,886 |
| 5 | Random Forest | 0,333 | 0,818 | 0,474 | 0,919 |
| 6 | Gradient Boosting | 0,407 | 0,500 | 0,449 | 0,893 |

**Modelo selecionado: XGBoost + SMOTE** — melhor F1 e melhor recall, identificando **21 dos 27** vinhos de alta qualidade do conjunto de teste. O tratamento do desbalanceamento (SMOTE) mostrou-se mais determinante para o desempenho do que a escolha do algoritmo em si.

---

## Interpretação

A importância das variáveis no modelo vencedor confirmou, por um caminho independente, os achados da EDA: as três variáveis mais influentes são as mesmas três mais correlacionadas com a qualidade.

| Posição | Variável | Importância |
|---|---|---|
| 1º | alcohol | 0,287 |
| 2º | sulphates | 0,108 |
| 3º | volatile acidity | 0,090 |

**Implicações para a produção:** vinhos de maior qualidade tendem a apresentar maior teor alcoólico (ligado à maturação da uva), maior concentração de sulfatos (conservação) e menor acidez volátil (indicador de deterioração). Esses fatores podem orientar o controle de qualidade ao longo do processo produtivo.

---

## Estrutura do repositório

```
wine-quality-classification/
│
├── data/              # Base de dados utilizada
├── notebooks/         # Notebook com a análise e modelagem
├── src/               # Scripts auxiliares (pré-processamento ou modelagem)
├── results/           # Gráficos e métricas dos modelos
├── requirements.txt   # Bibliotecas utilizadas
└── README.md          # Descrição do projeto
```

---

## Tecnologias

Python · pandas · scikit-learn · XGBoost · imbalanced-learn (SMOTE) · Plotly · Matplotlib · Seaborn · kagglehub

---

## Autores

Projeto desenvolvido para o Tech Challenge — Fase 2 (POSTECH).

Felipe Carnevale Fornaziero
