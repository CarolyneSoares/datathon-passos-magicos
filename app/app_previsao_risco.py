"""
🎓 App de Previsão de Risco de Defasagem Escolar
Desenvolvido com Streamlit + Random Forest (PKL)

Como executar:
    pip install streamlit scikit-learn pandas numpy joblib
    streamlit run app_previsao_risco.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ──────────────────────────────────────────────
# CONFIG DA PÁGINA
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Previsão de Risco — Defasagem Escolar",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CSS CUSTOMIZADO
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --verde:  #00C9A7;
    --amarelo: #FFD166;
    --vermelho: #EF233C;
    --azul: #0A1628;
    --cinza: #1E2D40;
    --cinza-claro: #2E3F54;
    --texto: #E8EFF7;
    --subtexto: #8FA3BF;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--azul);
    color: var(--texto);
}

/* Sidebar — fundo e texto geral */
[data-testid="stSidebar"] {
    background: var(--cinza) !important;
    border-right: 1px solid var(--cinza-claro);
}

/* Todos os textos dentro da sidebar em branco */
[data-testid="stSidebar"] * {
    color: var(--texto) !important;
}

/* Labels e captions da sidebar */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: var(--texto) !important;
}

/* File uploader texto */
[data-testid="stSidebar"] [data-testid="stFileUploader"] * {
    color: var(--texto) !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
    background: var(--cinza-claro) !important;
    border: 1px dashed rgba(143,163,191,0.4) !important;
    border-radius: 10px !important;
}

/* Botão Browse files */
[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    background: var(--cinza-claro) !important;
    color: var(--texto) !important;
    border: 1px solid rgba(143,163,191,0.3) !important;
    border-radius: 8px !important;
}

/* Warning e success boxes — texto branco */
[data-testid="stSidebar"] [data-testid="stAlert"] * {
    color: var(--texto) !important;
}
[data-testid="stSidebar"] [data-testid="stNotification"] * {
    color: var(--texto) !important;
}
[data-testid="stSidebar"] .stSuccess,
[data-testid="stSidebar"] .stSuccess *,
[data-testid="stSidebar"] .stWarning,
[data-testid="stSidebar"] .stWarning *,
[data-testid="stSidebar"] [data-baseweb="notification"] *,
[data-testid="stSidebar"] [role="alert"] * {
    color: var(--texto) !important;
}
/* Código inline dentro dos alertas (random_forest_model.pkl) */
[data-testid="stSidebar"] code {
    color: var(--verde) !important;
    background: rgba(0,201,167,0.15) !important;
    border-radius: 4px;
    padding: 1px 5px;
}

/* Markdown dentro da sidebar */
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown span,
[data-testid="stSidebar"] .stMarkdown div {
    color: var(--texto) !important;
}

/* Headers */
h1, h2, h3 {
    font-family: 'Space Mono', monospace;
    letter-spacing: -0.5px;
}

/* Cards de métricas */
.metric-card {
    background: var(--cinza);
    border: 1px solid var(--cinza-claro);
    border-radius: 16px;
    padding: 24px 28px;
    text-align: center;
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-2px); }

/* Barra de probabilidade */
.prob-bar-wrap {
    background: var(--cinza-claro);
    border-radius: 100px;
    height: 14px;
    width: 100%;
    overflow: hidden;
    margin: 8px 0 4px;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 0.6s ease;
}

/* Alerta de risco */
.alert-risco {
    background: #080D18;
    border: 1px solid rgba(239,35,60,0.5);
    border-left: 4px solid var(--vermelho);
    border-radius: 12px;
    padding: 20px 24px;
    margin: 16px 0;
}
.alert-seguro {
    background: #080D18;
    border: 1px solid rgba(0,201,167,0.5);
    border-left: 4px solid var(--verde);
    border-radius: 12px;
    padding: 20px 24px;
    margin: 16px 0;
}
.alert-alerta {
    background: #080D18;
    border: 1px solid rgba(255,209,102,0.5);
    border-left: 4px solid var(--amarelo);
    border-radius: 12px;
    padding: 20px 24px;
    margin: 16px 0;
}

/* Inputs */
[data-testid="stNumberInput"] > div > div > input,
[data-testid="stSelectbox"] > div > div,
[data-testid="stTextInput"] > div > div > input {
    background: var(--cinza-claro) !important;
    border: 1px solid rgba(143,163,191,0.25) !important;
    border-radius: 10px !important;
    color: var(--texto) !important;
}

/* Botão principal */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #00C9A7, #00A882) !important;
    color: #0A1628 !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 32px !important;
    width: 100% !important;
    letter-spacing: 0.5px;
    transition: all 0.2s !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(0,201,167,0.35) !important;
}

/* Labels */
label, .stSelectbox label, .stNumberInput label {
    color: var(--subtexto) !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}

/* Divider */
hr { border-color: var(--cinza-claro); }

/* Gauge container */
.gauge-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 32px 0;
}

/* Chip de status */
.chip {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 100px;
    font-size: 12px;
    font-weight: 600;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.5px;
}
.chip-risco { background: rgba(239,35,60,0.2); color: #ff6b7a; border: 1px solid rgba(239,35,60,0.4); }
.chip-alerta { background: rgba(255,209,102,0.2); color: #ffd166; border: 1px solid rgba(255,209,102,0.4); }
.chip-seguro { background: rgba(0,201,167,0.2); color: #00c9a7; border: 1px solid rgba(0,201,167,0.4); }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# CARREGAR MODELO
# ──────────────────────────────────────────────
@st.cache_resource
def carregar_modelo(path: str):
    return joblib.load(path)
 
 
def encontrar_modelo():
    """Procura o arquivo PKL no diretório atual, pasta modelo/ e diretórios comuns."""
    base = os.path.dirname(__file__)
    candidatos = [
        # mesma pasta do app
        os.path.join(base, "random_forest_model.pkl"),
        # pasta modelo/ ao lado do app
        os.path.join(base, "modelo", "random_forest_model.pkl"),
        # pasta modelo/ um nível acima (raiz do projeto)
        os.path.join(base, "..", "modelo", "random_forest_model.pkl"),
        # caminho relativo simples
        "random_forest_model.pkl",
        os.path.join("modelo", "random_forest_model.pkl"),
    ]
    for p in candidatos:
        if os.path.exists(p):
            return os.path.normpath(p)
    return None


# ──────────────────────────────────────────────
# HEADER PRINCIPAL
# ──────────────────────────────────────────────
col_logo, col_titulo = st.columns([1, 6])
with col_logo:
    st.markdown("<div style='padding-top: 8px'>", unsafe_allow_html=True)
    base = os.path.dirname(__file__)
    logo_candidatos = [
        os.path.join(base, "Logo", "PassosMagicos.png"),
        os.path.join(base, "..", "Logo", "PassosMagicos.png"),
        os.path.join("Logo", "PassosMagicos.png"),
        os.path.join(base, "PassosMagicos.png"),
        "PassosMagicos.png",
    ]
    logo_encontrado = next((p for p in logo_candidatos if os.path.exists(p)), None)
    if logo_encontrado:
        st.image(logo_encontrado, width=110)
    st.markdown("</div>", unsafe_allow_html=True)
with col_titulo:
    st.markdown("""
    <div style='padding: 8px 0 4px'>
        <div style='font-family: Space Mono, monospace; font-size: 11px; color: #8FA3BF; letter-spacing: 3px; text-transform: uppercase; margin-bottom: 6px'>
            Machine Learning · Random Forest Classifier
        </div>
        <h1 style='margin: 0; font-size: 32px; background: linear-gradient(90deg, #00C9A7, #4facfe); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            Previsão de Risco de Defasagem
        </h1>
        <p style='color: #8FA3BF; margin: 6px 0 0; font-size: 15px;'>
            Insira os dados do(a) aluno(a) para calcular a probabilidade de risco de defasagem escolar.
        </p>
    </div>
    """, unsafe_allow_html=True)
 
st.markdown("<hr style='margin: 20px 0 28px'>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# SIDEBAR — CARREGAMENTO DO MODELO
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family: Space Mono, monospace; font-size: 11px; color: #8FA3BF; letter-spacing: 2px; margin-bottom: 12px'>
        ⚙️ CONFIGURAÇÃO DO MODELO
    </div>
    """, unsafe_allow_html=True)

    caminho_auto = encontrar_modelo()
    if caminho_auto:
        st.success(f"✅ Modelo encontrado automaticamente:\n`{os.path.basename(caminho_auto)}`")
        modelo_path = caminho_auto
    else:
        st.warning("Modelo PKL não encontrado no diretório atual.")
        modelo_path = None

    arquivo_upload = st.file_uploader(
        "Ou carregue o modelo PKL aqui:",
        type=["pkl"],
        help="Selecione o arquivo random_forest_model.pkl"
    )

    if arquivo_upload:
        tmp_path = "/tmp/modelo_carregado.pkl"
        with open(tmp_path, "wb") as f:
            f.write(arquivo_upload.read())
        modelo_path = tmp_path
        st.success("✅ Modelo carregado com sucesso!")

    st.markdown("---")
    st.markdown("""
    <div style='font-family: Space Mono, monospace; font-size: 11px; color: #8FA3BF; letter-spacing: 2px; margin-bottom: 10px'>
        📊 LEGENDA DE RISCO
    </div>
    <div style='font-size: 13px; line-height: 2'>
        <span class='chip chip-seguro'>BAIXO</span>&nbsp; < 35% de probabilidade<br>
        <span class='chip chip-alerta'>MÉDIO</span>&nbsp; 35% – 65%<br>
        <span class='chip chip-risco'>ALTO</span>&nbsp; > 65%<br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Modelo: RandomForestClassifier  \nAcurácia (teste): ~80.9%  \nVariável-alvo: `risco_defasagem`")


# ──────────────────────────────────────────────
# FORMULÁRIO DE ENTRADA
# ──────────────────────────────────────────────
st.markdown("""
<div style='font-family: Space Mono, monospace; font-size: 11px; color: #8FA3BF; letter-spacing: 2px; margin-bottom: 18px'>
    📋 DADOS DO(A) ALUNO(A)
</div>
""", unsafe_allow_html=True)

with st.form("formulario_aluno"):
    # ── BLOCO 1: NOTAS ──
    st.markdown("#### 📚 Desempenho Acadêmico")
    c1, c2, c3 = st.columns(3)
    with c1:
        mat = st.number_input("Matemática (Mat)", min_value=0.0, max_value=10.0, value=6.5, step=0.1,
                              help="Nota de Matemática (0–10)")
    with c2:
        por = st.number_input("Português (Por)", min_value=0.0, max_value=10.0, value=6.5, step=0.1,
                              help="Nota de Português (0–10)")
    with c3:
        ing = st.number_input("Inglês (Ing)", min_value=0.0, max_value=10.0, value=6.0, step=0.1,
                              help="Nota de Inglês (0–10)")

    st.markdown("<div style='margin:8px 0'></div>", unsafe_allow_html=True)

    # ── BLOCO 2: INDICADORES ──
    st.markdown("#### 📈 Indicadores PEDE")
    c4, c5, c6, c7 = st.columns(4)
    with c4:
        ida = st.number_input("IDA", min_value=0.0, max_value=10.0, value=6.0, step=0.1,
                               help="Índice de Desempenho Acadêmico")
    with c5:
        ieg = st.number_input("IEG", min_value=0.0, max_value=10.0, value=6.5, step=0.1,
                               help="Índice de Engajamento")
    with c6:
        ips = st.number_input("IPS", min_value=0.0, max_value=10.0, value=6.5, step=0.1,
                               help="Índice Psicossocial")
    with c7:
        ipp = st.number_input("IPP", min_value=0.0, max_value=10.0, value=6.5, step=0.1,
                               help="Índice de Participação e Proatividade")

    c8, c9, _, __ = st.columns(4)
    with c8:
        ian = st.number_input("IAN", min_value=0.0, max_value=10.0, value=6.5, step=0.1,
                               help="Índice de Adequação ao Nível")
    with c9:
        inde = st.number_input("INDE (apenas informativo)", min_value=0.0, max_value=10.0, value=6.5, step=0.1,
                                help="Índice de Desenvolvimento Educacional — não entra no modelo, apenas referência")

    st.markdown("<div style='margin:8px 0'></div>", unsafe_allow_html=True)
    
    #Lista de instituições para o selectbox
    lista_instituicoes = lista_instituicoes = [
    'Pública', 
    'Rede Decisão',
    'Privada - Programa de Apadrinhamento',
    'Privada',
    'Escola Pública',
    'Escola JP II',
    'Concluiu o 3º EM'
]
    # ── BLOCO 3: INFORMAÇÕES DO ALUNO ──
    st.markdown("#### 🧑‍🎓 Informações do(a) Aluno(a)")
    c10, c11, c12, c13 = st.columns(4)
    with c10:
        fase = st.number_input("Fase atual", min_value=0, max_value=8, value=3, step=1,
                                help="Fase atual do aluno (0 a 8)")
    with c11:
        fase_ideal = st.number_input("Fase Ideal", min_value=0, max_value=8, value=3, step=1,
                                      help="Fase ideal para a idade do aluno")
    with c12:
        idade = st.number_input("Idade", min_value=6, max_value=25, value=12, step=1)
    with c13:
        ano_ingresso = st.number_input("Ano de Ingresso", min_value=2000, max_value=2030, value=2020, step=1)

    c14, c15, c16 = st.columns(3)
    with c14:
        ano = st.number_input("Ano de referência", min_value=2000, max_value=2030, value=2023, step=1,
                               help="Ano de referência para a avaliação (normalmente o ano atual)")
    with c15:
        genero = st.selectbox("Gênero", options=["Feminino", "Masculino"], index=0)
    with c16:
        instituicao = st.selectbox("Instituição de Ensino", lista_instituicoes)

    turma = st.text_input("Turma", value="7A", help="Código da turma (ex: 7A, 8B)")

    st.markdown("<div style='margin:16px 0'></div>", unsafe_allow_html=True)
    submit = st.form_submit_button("🔍  CALCULAR RISCO DE DEFASAGEM")


# ──────────────────────────────────────────────
# RESULTADO
# ──────────────────────────────────────────────
if submit:
    if not modelo_path:
        st.error("❌ Nenhum modelo PKL foi encontrado. Faça o upload na barra lateral.")
        st.stop()

    try:
        model = carregar_modelo(modelo_path)
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        st.stop()

    # Montar DataFrame de entrada (mesma estrutura usada no treino)
    entrada = pd.DataFrame([{
        "Mat": mat,
        "Por": por,
        "Ing": ing,
        "IDA": ida,
        "IEG": ieg,
        "IPS": ips,
        "IPP": ipp,
        "IAN": ian,
        "Fase": fase,
        "Fase Ideal": fase_ideal,
        "Idade": idade,
        "Ano ingresso": ano_ingresso,
        "ano": ano,
        "Instituição de ensino": instituicao,
        "Turma": turma,
        "Gênero": genero,
    }])

    # Predição
    try:
        prob_classes = model.predict_proba(entrada)[0]
        # Classe 0 = sem risco, Classe 1 = com risco
        if len(prob_classes) == 2:
            prob_risco = prob_classes[1]
            prob_seguro = prob_classes[0]
        else:
            prob_risco = prob_classes[-1]
            prob_seguro = 1 - prob_risco

        predicao = model.predict(entrada)[0]
    except Exception as e:
        st.error(f"Erro ao fazer predição: {e}")
        st.stop()

    # Classificação de nível de risco
    if prob_risco < 0.35:
        nivel = "BAIXO"
        chip_class = "chip-seguro"
        cor = "#00C9A7"
        emoji = "✅"
        mensagem = "O(a) aluno(a) apresenta baixo risco de defasagem. Continue monitorando o desempenho regularmente."
        alert_class = "alert-seguro"
    elif prob_risco < 0.65:
        nivel = "MÉDIO"
        chip_class = "chip-alerta"
        cor = "#FFD166"
        emoji = "⚠️"
        mensagem = "Atenção: risco moderado identificado. Recomenda-se acompanhamento pedagógico preventivo e reunião com a família."
        alert_class = "alert-alerta"
    else:
        nivel = "ALTO"
        chip_class = "chip-risco"
        cor = "#EF233C"
        emoji = "🚨"
        mensagem = "Risco elevado de defasagem detectado! Intervenção pedagógica imediata é recomendada. Acione o suporte psicossocial e pedagógico."
        alert_class = "alert-risco"

    prob_pct = prob_risco * 100
    seguro_pct = prob_seguro * 100

    st.markdown("<hr style='margin: 24px 0'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family: Space Mono, monospace; font-size: 11px; color: #8FA3BF; letter-spacing: 2px; margin-bottom: 18px'>
        🎯 RESULTADO DA ANÁLISE
    </div>
    """, unsafe_allow_html=True)

    # Linha de resultados
    col_gauge, col_detalhe = st.columns([2, 3])

    with col_gauge:
        st.markdown(f"""
        <div class='metric-card' style='border-color: {cor}60; background: #080D18;'>
            <div style='font-family: Space Mono, monospace; font-size: 11px; color: #8FA3BF; letter-spacing: 2px; margin-bottom: 8px'>PROBABILIDADE DE RISCO</div>
            <div style='font-size: 72px; font-family: Space Mono, monospace; font-weight: 700; color: {cor}; line-height: 1'>
                {prob_pct:.1f}<span style='font-size: 32px'>%</span>
            </div>
            <div style='margin: 16px 0 8px'>
                <div class='prob-bar-wrap'>
                    <div class='prob-bar-fill' style='width: {prob_pct}%; background: linear-gradient(90deg, {cor}AA, {cor})'></div>
                </div>
            </div>
            <div style='margin-top: 14px'>
                <span class='chip {chip_class}'>{emoji} RISCO {nivel}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_detalhe:
        st.markdown(f"""
        <div class='{alert_class}'>
            <div style='font-family: Space Mono, monospace; font-size: 12px; color: {cor}; margin-bottom: 8px; letter-spacing: 1px'>
                {emoji} DIAGNÓSTICO
            </div>
            <p style='margin: 0; font-size: 15px; line-height: 1.6; color: var(--texto)'>{mensagem}</p>
        </div>
        """, unsafe_allow_html=True)

        # Breakdown probabilidades
        st.markdown(f"""
        <div style='margin-top: 16px; display: flex; gap: 12px'>
            <div class='metric-card' style='flex: 1; padding: 16px; background: #080D18; border-color: rgba(0,201,167,0.3)'>
                <div style='font-size: 11px; color: #8FA3BF; letter-spacing: 1.5px; margin-bottom: 6px'>SEM RISCO</div>
                <div style='font-size: 28px; font-family: Space Mono, monospace; font-weight: 700; color: #00C9A7'>{seguro_pct:.1f}%</div>
            </div>
            <div class='metric-card' style='flex: 1; padding: 16px; background: #080D18; border-color: rgba(239,35,60,0.3)'>
                <div style='font-size: 11px; color: #8FA3BF; letter-spacing: 1.5px; margin-bottom: 6px'>EM RISCO</div>
                <div style='font-size: 28px; font-family: Space Mono, monospace; font-weight: 700; color: #EF233C'>{prob_pct:.1f}%</div>
            </div>
            <div class='metric-card' style='flex: 1; padding: 16px; background: #080D18; border-color: rgba(143,163,191,0.3)'>
                <div style='font-size: 11px; color: #8FA3BF; letter-spacing: 1.5px; margin-bottom: 6px'>DEFASAGEM ATUAL</div>
                <div style='font-size: 28px; font-family: Space Mono, monospace; font-weight: 700; color: {"#EF233C" if fase < fase_ideal else "#00C9A7"}'>{fase - fase_ideal:+d}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Tabela de inputs
    st.markdown("<div style='margin: 28px 0 16px'></div>", unsafe_allow_html=True)
    with st.expander("📋 Ver dados inseridos"):
        df_exibir = entrada.copy()
        df_exibir.index = ["Aluno(a)"]
        st.dataframe(df_exibir.style.set_properties(**{
            'background-color': '#1E2D40',
            'color': '#E8EFF7',
            'border': '1px solid #2E3F54'
        }), use_container_width=True)

    # Recomendações
    st.markdown("<div style='margin: 16px 0'></div>", unsafe_allow_html=True)
    with st.expander("💡 Recomendações pedagógicas"):
        if nivel == "BAIXO":
            st.markdown("""
            - ✅ Manter o acompanhamento regular do desempenho
            - ✅ Estimular participação em atividades extracurriculares
            - ✅ Registrar progresso no prontuário do aluno
            """)
        elif nivel == "MÉDIO":
            st.markdown("""
            - ⚠️ Agendar reunião com professores e família nas próximas 2 semanas
            - ⚠️ Monitorar a frequência escolar com atenção redobrada
            - ⚠️ Verificar dificuldades específicas em Matemática e Português
            - ⚠️ Considerar reforço escolar nas disciplinas com notas abaixo de 6.0
            """)
        else:
            st.markdown("""
            - 🚨 **Ação imediata**: Convocar reunião urgente com família e equipe pedagógica
            - 🚨 Acionar o time de psicologia / assistência social se necessário
            - 🚨 Elaborar Plano de Intervenção Individual (PII)
            - 🚨 Reforço escolar intensivo em todas as disciplinas críticas
            - 🚨 Revisão do enquadramento de fase — possível necessidade de reclassificação
            """)

# ──────────────────────────────────────────────
# RODAPÉ
# ──────────────────────────────────────────────
st.markdown("<div style='margin-top: 48px'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; border-top: 1px solid #1E2D40; padding: 24px 0 16px;'>
    <div style='color: #8FA3BF; font-size: 12px; margin-bottom: 10px; font-family: Space Mono, monospace; letter-spacing: 1px;'>
        Previsão de Risco de Defasagem &nbsp;·&nbsp; Random Forest Classifier &nbsp;·&nbsp; Datathon FIAP Fase 5
    </div>
    <div style='color: #4A6080; font-size: 12px;'>
        Desenvolvido por &nbsp;
        <span style='color: #8FA3BF;'>Carolyne Rafaella Soares Costa</span>
        <span style='color: #4A6080;'> (RM361016)</span>
        &nbsp;·&nbsp;
        <span style='color: #8FA3BF;'>Natane Cezario Barbosa</span>
        <span style='color: #4A6080;'> (RM362783)</span>
    </div>
</div>
""", unsafe_allow_html=True)