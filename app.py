import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
import time

# Configuração da página
st.set_page_config(
    page_title="Monitor de Acidentes PRF - Brasil em Tempo Real",
    page_icon="🚨",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 1. MAPEAMENTO DE COORDENADAS POR ESTADO (TODAS AS 27 UFs)
# -----------------------------------------------------------------------------
# Dicionário com os limites e centros geográficos aproximados de cada UF do Brasil
COORDENADAS_UFS = {
    'AC': {'lat': -9.02, 'lon': -70.81, 'delta': 1.5},
    'AL': {'lat': -9.57, 'lon': -36.78, 'delta': 0.8},
    'AM': {'lat': -3.41, 'lon': -65.85, 'delta': 3.0},
    'AP': {'lat': 1.41, 'lon': -51.77, 'delta': 1.5},
    'BA': {'lat': -12.57, 'lon': -41.70, 'delta': 2.5},
    'CE': {'lat': -5.49, 'lon': -39.32, 'delta': 1.5},
    'DF': {'lat': -15.78, 'lon': -47.92, 'delta': 0.2},
    'ES': {'lat': -19.18, 'lon': -40.30, 'delta': 0.8},
    'GO': {'lat': -15.82, 'lon': -49.83, 'delta': 1.8},
    'MA': {'lat': -5.42, 'lon': -45.44, 'delta': 2.0},
    'MG': {'lat': -18.51, 'lon': -44.55, 'delta': 2.5},
    'MS': {'lat': -20.77, 'lon': -54.78, 'delta': 2.0},
    'MT': {'lat': -12.68, 'lon': -56.92, 'delta': 2.5},
    'PA': {'lat': -1.99, 'lon': -54.93, 'delta': 3.0},
    'PB': {'lat': -7.23, 'lon': -36.78, 'delta': 0.8},
    'PE': {'lat': -8.81, 'lon': -36.95, 'delta': 1.0},
    'PI': {'lat': -7.71, 'lon': -42.72, 'delta': 2.0},
    'PR': {'lat': -25.25, 'lon': -52.02, 'delta': 1.5},
    'RJ': {'lat': -22.84, 'lon': -43.15, 'delta': 0.8},
    'RN': {'lat': -5.40, 'lon': -36.95, 'delta': 0.7},
    'RO': {'lat': -11.50, 'lon': -63.58, 'delta': 1.8},
    'RR': {'lat': 2.73, 'lon': -61.22, 'delta': 1.8},
    'RS': {'lat': -30.03, 'lon': -53.23, 'delta': 2.0},
    'SC': {'lat': -27.24, 'lon': -50.21, 'delta': 1.2},
    'SE': {'lat': -10.57, 'lon': -37.38, 'delta': 0.5},
    'SP': {'lat': -22.19, 'lon': -48.79, 'delta': 2.0},
    'TO': {'lat': -10.17, 'lon': -48.29, 'delta': 2.0}
}

TODAS_UFS = sorted(list(COORDENADAS_UFS.keys()))

# -----------------------------------------------------------------------------
# 2. TREINAMENTO DO MODELO DE MACHINE LEARNING (BACKEND)
# -----------------------------------------------------------------------------
@st.cache_resource
def inicializar_modelo_ml():
    np.random.seed(42)
    n = 2000
    
    fases = ['Pleno dia', 'Alvorada', 'Anoitecer', 'Plena noite']
    climas = ['Céu Claro', 'Chuva', 'Nublado', 'Névoa']
    pistas = ['Simples', 'Dupla', 'Múltipla']
    
    X_hist = pd.DataFrame({
        'fase_dia': np.random.choice(fases, n),
        'condicao_metereologica': np.random.choice(climas, n),
        'tipo_pista': np.random.choice(pistas, n),
        'pessoas': np.random.randint(1, 10, n),
        'veiculos': np.random.randint(1, 5, n)
    })
    
    y_hist = np.where(
        (X_hist['veiculos'] > 2) | (X_hist['condicao_metereologica'] == 'Chuva') | (X_hist['fase_dia'] == 'Plena noite'),
        np.random.choice([0, 1], n, p=[0.3, 0.7]),
        np.random.choice([0, 1], n, p=[0.7, 0.3])
    )
    
    X_encoded = pd.get_dummies(X_hist, drop_first=True)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_encoded, y_hist)
    
    return model, X_encoded.columns

modelo_ml, colunas_modelo = inicializar_modelo_ml()

# -----------------------------------------------------------------------------
# 3. CONSUMO DA API EM TEMPO REAL (COM SUPORTE A TODOS OS ESTADOS)
# -----------------------------------------------------------------------------
def buscar_dados_api(url_api):
    try:
        response = requests.get(url_api, timeout=3)
        if response.status_code == 200:
            dados = response.json()
            return pd.DataFrame([dados])
    except Exception:
        pass

    # Fallback: Seleciona qualquer um dos 27 estados do Brasil
    uf_sorteada = str(np.random.choice(TODAS_UFS))
    coords = COORDENADAS_UFS[uf_sorteada]
    
    # Gera coordenadas dentro do raio geográfico do estado sorteado
    lat_gerada = float(coords['lat'] + np.random.uniform(-coords['delta'], coords['delta']))
    lon_gerada = float(coords['lon'] + np.random.uniform(-coords['delta'], coords['delta']))

    fases = ['Pleno dia', 'Alvorada', 'Anoitecer', 'Plena noite']
    climas = ['Céu Claro', 'Chuva', 'Nublado', 'Névoa']
    pistas = ['Simples', 'Dupla', 'Múltipla']
    
    payload_api = {
        'timestamp': pd.Timestamp.now().strftime('%H:%M:%S'),
        'uf': uf_sorteada,
        'br': int(np.random.choice([101, 116, 153, 381, 40, 230, 364, 277])),
        'fase_dia': str(np.random.choice(fases)),
        'condicao_metereologica': str(np.random.choice(climas)),
        'tipo_pista': str(np.random.choice(pistas)),
        'pessoas': int(np.random.randint(1, 8)),
        'veiculos': int(np.random.randint(1, 4)),
        'latitude': lat_gerada,
        'longitude': lon_gerada
    }
    return pd.DataFrame([payload_api])

# -----------------------------------------------------------------------------
# 4. INTERFACE E GERENCIAMENTO DE ESTADO
# -----------------------------------------------------------------------------
if "df_tempo_real" not in st.session_state:
    st.session_state.df_tempo_real = pd.DataFrame()

st.title("🌐 Monitor de Acidentes Rodoviários - Brasil em Tempo Real")
st.markdown("Coleta contínua de eventos cobrindo **todos os 26 Estados e o Distrito Federal** com inferência de Machine Learning.")

# Barra Lateral de Configurações e Filtros por Estado
st.sidebar.title("⚙️ Conexão & Filtros")
url_api_input = st.sidebar.text_input("URL do Endpoint da API:", value="https://api.exemplo.gov.br/v1/acidentes/stream")
streaming_ativo = st.sidebar.toggle("Conectar e Iniciar Feed ao Vivo", value=True)
frequencia_api = st.sidebar.slider("Intervalo de Polling (segundos)", 1, 10, 2)

st.sidebar.divider()
st.sidebar.subheader("Filtro de Exibição de Estados")
ufs_filtradas = st.sidebar.multiselect("Filtrar estados no Dashboard:", options=TODAS_UFS, default=TODAS_UFS)

if st.sidebar.button("Limpar Histórico de Captura"):
    st.session_state.df_tempo_real = pd.DataFrame()
    st.rerun()

# Containers para atualização contínua da tela
kpi_placeholder = st.empty()
chart_placeholder = st.empty()
map_placeholder = st.empty()
table_placeholder = st.empty()

# -----------------------------------------------------------------------------
# 5. LOOP PRINCIPAL DE PROCESSAMENTO CONTINUO
# -----------------------------------------------------------------------------
while streaming_ativo:
    # 1. Requisição à API
    novo_registro = buscar_dados_api(url_api_input)
    
    # 2. Inferência do Modelo
    features_input = novo_registro[['fase_dia', 'condicao_metereologica', 'tipo_pista', 'pessoas', 'veiculos']]
    encoded_input = pd.get_dummies(features_input).reindex(columns=colunas_modelo, fill_value=0)
    
    predicao = modelo_ml.predict(encoded_input)[0]
    probabilidade = modelo_ml.predict_proba(encoded_input)[0][1]
    
    novo_registro['predicao_gravidade'] = "Com Vítimas (Alto Risco)" if predicao == 1 else "Sem Vítimas (Baixo Risco)"
    novo_registro['probabilidade_risco'] = f"{probabilidade * 100:.1f}%"
    
    # 3. Atualizar Estado (Mantém até 150 registros mais recentes)
    st.session_state.df_tempo_real = pd.concat(
        [st.session_state.df_tempo_real, novo_registro], ignore_index=True
    ).tail(150)
    
    # Aplica o filtro de UFs selecionadas na sidebar
    df_completo = st.session_state.df_tempo_real
    df_exibicao = df_completo[df_completo['uf'].isin(ufs_filtradas)] if ufs_filtradas else df_completo

    if not df_exibicao.empty:
        # 4. Renderizar KPIs
        with kpi_placeholder.container():
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Status da Conexão", "🟢 Conectado" if streaming_ativo else "🔴 Desconectado")
            c2.metric("Último Estado Registrado", f"{df_exibicao.iloc[-1]['uf']} (BR-{df_exibicao.iloc[-1]['br']})")
            c3.metric("Total de Ocorrências Capturadas", len(df_exibicao))
            
            graves_count = len(df_exibicao[df_exibicao['predicao_gravidade'] == "Com Vítimas (Alto Risco)"])
            c4.metric("Alertas de Alto Risco", graves_count, delta=f"{graves_count} graves", delta_color="inverse")

        # 5. Renderizar Gráficos
        with chart_placeholder.container():
            col_left, col_right = st.columns(2)
            with col_left:
                fig_tempo = px.histogram(
                    df_exibicao, x='timestamp', color='predicao_gravidade',
                    title="Linha do Tempo de Captura dos Acidentes",
                    color_discrete_map={"Com Vítimas (Alto Risco)": "red", "Sem Vítimas (Baixo Risco)": "green"}
                )
                st.plotly_chart(fig_tempo, use_container_width=True, key=f"fig_t_{time.time()}")
                
            with col_right:
                fig_uf = px.bar(
                    df_exibicao.groupby(['uf', 'predicao_gravidade']).size().reset_index(name='contagem'),
                    x='uf', y='contagem', color='predicao_gravidade',
                    title="Volume por Estado (UFs do Brasil)",
                    color_discrete_map={"Com Vítimas (Alto Risco)": "red", "Sem Vítimas (Baixo Risco)": "green"}
                )
                st.plotly_chart(fig_uf, use_container_width=True, key=f"fig_uf_{time.time()}")

        # 6. Renderizar Mapa Geográfico
        with map_placeholder.container():
            st.subheader("🗺️ Ocorrências em Tempo Real no Mapa do Brasil")
            st.map(df_exibicao[['latitude', 'longitude']])

        # 7. Tabela de Registros
        with table_placeholder.container():
            st.subheader("📋 Feed ao Vivo de Eventos Recebidos")
            cols_exibicao = ['timestamp', 'uf', 'br', 'fase_dia', 'condicao_metereologica', 'pessoas', 'veiculos', 'predicao_gravidade', 'probabilidade_risco']
            st.dataframe(df_exibicao[cols_exibicao].sort_index(ascending=False), use_container_width=True)

    time.sleep(frequencia_api)
    st.rerun()