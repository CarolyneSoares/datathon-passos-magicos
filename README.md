# 🎓 Datathon — Associação Passos Mágicos

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=flat-square&logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Concluído-green?style=flat-square)

> Projeto desenvolvido para o **Datathon FIAP — Fase 5**, em parceria com a **Associação Passos Mágicos**, com o objetivo de prever o risco de defasagem escolar de alunos utilizando Machine Learning.

---

## 📌 Sobre o Projeto

A Associação Passos Mágicos tem uma trajetória de 32 anos de atuação, trabalhando na transformação da vida de crianças e jovens de baixa renda no município de Embu-Guaçu. Este projeto utiliza os dados do programa **PEDE (2022–2024)** para construir um modelo preditivo capaz de identificar alunos com risco de defasagem escolar, permitindo intervenções pedagógicas antecipadas.

---

## ❓ Perguntas de Negócio Respondidas

| # | Tema | Pergunta |
|---|------|----------|
| 1 | **IAN — Adequação do Nível** | Qual é o perfil geral de defasagem dos alunos e como ele evolui ao longo do ano? |
| 2 | **IDA — Desempenho Acadêmico** | O desempenho acadêmico médio está melhorando, estagnado ou caindo ao longo das fases e anos? |
| 3 | **IEG — Engajamento** | O grau de engajamento tem relação direta com os indicadores de desempenho (IDA) e ponto de virada (IPV)? |
| 4 | **IAA — Autoavaliação** | As percepções dos alunos sobre si mesmos são coerentes com seu desempenho real e engajamento? |
| 5 | **IPS — Aspectos Psicossociais** | Há padrões psicossociais que antecedem quedas de desempenho ou de engajamento? |
| 6 | **IPP — Aspectos Psicopedagógicos** | As avaliações psicopedagógicas confirmam ou contradizem a defasagem identificada pelo IAN? |
| 7 | **IPV — Ponto de Virada** | Quais comportamentos acadêmicos, emocionais ou de engajamento mais influenciam o IPV ao longo do tempo? |
| 8 | **Multidimensionalidade** | Quais combinações de indicadores (IDA + IEG + IPS + IPP) elevam mais a nota global do aluno (INDE)? |
| 9 | **Machine Learning — Risco** | Quais padrões nos indicadores permitem identificar alunos em risco antes de queda no desempenho? |
| 10 | **Efetividade do Programa** | Os indicadores mostram melhora consistente ao longo do ciclo nas diferentes fases (Quartzo, Ágata, Ametista e Topázio)? |
| 11 | **Insights & Criatividade** | Sugestões e pontos de vista adicionais com base na análise exploratória dos dados. |

---

## 🗂️ Estrutura do Repositório

```
datathon-passos-magicos/
│
├── app/
│   ├── app_previsao_risco.py                        # App Streamlit de previsão de risco
│   └── requirements.txt                              # Dependências do projeto
│
├── data/
│   └── BASE DE DADOS PEDE 2024 - DATATHON.xlsx      # Base de dados (não versionada no Git)
│
├── graficos/                                         # Gráficos gerados na análise exploratória
│   ├── correlacao_autoavaliacao.png
│   ├── distribuicao_ian.png
│   ├── Distribuição_ida.png
│   ├── evolucao_ian_categorias.png
│   ├── evolucao_ida_categorias.png
│   ├── gradiente_autoavaliacao_barras.png
│   ├── gradiente_engajamento_barras.png
│   ├── heatmap_correlacao_engajamento.png
│   ├── ida_por_autoavaliacao.png
│   ├── ida_por_engajamento.png
│   ├── ieg_por_autoavaliacao.png
│   ├── ipv_por_engajamento.png
│   ├── media_ian_ipp.png
│   ├── p5_gradiente_psicossocial_barras.png
│   ├── p5_prob_queda_ida.png
│   ├── p5_prob_queda_ieg.png
│   ├── p6_boxplot_ipp_por_ian.png
│   ├── p6_correlacao_ian_ipp.png
│   ├── p7_correlacao_ipv.png
│   ├── p7_heatmap_ipv_tempo.png
│   ├── p7_importancia_ipv.png
│   └── p10_barras_por_pedra_em_cada_ano.png
│
├── Logo/
│   └── PassosMagicos.png                             # Logo da associação
│
├── modelo/                                           # Modelos treinados
│   ├── catboost_model.pkl
│   ├── lightgbm_model.pkl
│   ├── random_forest_model.pkl                       # ✅ Modelo usado no app
│   ├── regressao_logistica_model.pkl
│   ├── svm_model.pkl
│   └── xgboost_model.pkl
│
├── notebook/
│   └── Datathon – Associação Passos Mágicos.ipynb   # Análise completa + gráficos + modelos
│
├── slide-apresentacao/
│   └── Slide Datathon – Associação Passos Mágicos.pptx
│
└── README.md
```

---

## 🤖 Modelos Treinados

| # | Modelo              | Acurácia Treino | Acurácia Teste | Diferença | Observação               |
|---|---------------------|----------------|---------------|-----------|--------------------------|
| 0 | Regressão Logística | 82.34%         | 79.30%        | 0.0304    | ✅ Modelo generaliza bem  |
| 1 | SVM                 | 85.53%         | 79.69%        | 0.0584    | ✅ Modelo generaliza bem  |
| 2 | **Random Forest**   | **83.73%**     | **80.86%**    | **0.0287**| ✅ **Modelo escolhido**   |
| 3 | XGBoost             | 98.00%         | 85.55%        | 0.1246    | ⚠️ Possível Overfitting   |
| 4 | CatBoost            | 93.61%         | 84.38%        | 0.0924    | ⚠️ Possível Overfitting   |
| 5 | LightGBM            | 95.11%         | 87.11%        | 0.0800    | ⚠️ Possível Overfitting   |

> O **Random Forest** foi escolhido para o app por apresentar a **menor diferença entre treino e teste (0.0287)**, garantindo melhor generalização sem overfitting.

---

## 🚀 Como Rodar o App

**1. Clone o repositório**
```bash
git clone https://github.com/seu-usuario/datathon-passos-magicos.git
cd datathon-passos-magicos/app
```

**2. Instale as dependências**
```bash
pip install -r requirements.txt
```

**3. Execute o app**
```bash
streamlit run app_previsao_risco.py
```

O app abrirá automaticamente em `http://localhost:8501`

---

## 🔍 Funcionalidades do App

- 📋 Formulário completo com os dados do(a) aluno(a)
- 📊 Probabilidade de risco exibida com barra visual
- 🚦 Classificação em 3 níveis: **BAIXO** / **MÉDIO** / **ALTO**
- 💡 Recomendações pedagógicas personalizadas por nível de risco
- 📁 Upload manual do modelo PKL pela barra lateral

---

## 📊 Variável-Alvo

| Valor | Significado |
|-------|-------------|
| `0`   | Sem risco de defasagem |
| `1`   | Em risco de defasagem  |

> Definição: `risco_defasagem = 1` quando a **Defasagem futura < 0** (aluno abaixo da fase ideal).

---

## 🧩 Features Utilizadas no Modelo

| Tipo        | Variáveis |
|-------------|-----------|
| Notas       | Mat, Por, Ing |
| Indicadores | IDA, IEG, IPS, IPP, IAN |
| Perfil      | Fase, Fase Ideal, Idade, Ano ingresso, Ano |
| Categóricas | Gênero, Turma, Instituição de ensino |

---

## ✅ Checklist de Entrega

- [x] Repositório GitHub com códigos de limpeza e análise
- [x] Notebook Python com modelo preditivo (feature engineering, treino/teste, avaliação)
- [x] Aplicação Streamlit com deploy no Community Cloud
- [x] Apresentação em formato PPT com storytelling
- [x] Vídeo de até 5 minutos apresentando os resultados

---

## 👩‍💻 Desenvolvedoras

| Nome | RM |
|------|----|
| Carolyne Rafaella Soares Costa | RM361016 |
| Natane Cezario Barbosa | RM362783 |

---

## 🏫 Instituição

**FIAP** — Faculdade de Informática e Administração Paulista  
Pós-Tech · Data Analytics · Fase 5 — Datathon
