import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Dashboard Principal - FCARE", page_icon="📊", layout="wide")

st.title("📊 Dashboard Principal - FCARE")
st.write("Visão geral das transacções e estatísticas.")

if "dataset" in st.session_state:
    df = st.session_state["dataset"]
else:
    st.warning("Nenhum dataset carregado. Volte à página inicial e carregue o arquivo.")
    st.stop()

# Criando coluna "estado"
df["estado"] = df["classe"].apply(lambda x: "Fraudulenta" if x == 1 else "Legítima")
df = st.session_state.get("dataset")
modelo = st.session_state.get("modelo")

# 🔹 Indicadores
col1, col2, col3 = st.columns(3)
col1.metric("Total de Transacções", len(df))
col2.metric("Fraudes Detectadas", (df["estado"] == "Fraudulenta").sum())
taxa_fraude = (df["estado"] == "Fraudulenta").mean() * 100
col3.metric("Taxa de Fraude (%)", f"{taxa_fraude:.2f}%")


st.markdown("---")

# 🔹 Tabela de transações
st.subheader("📄 Tabela de Transacções")
st.dataframe(df)

# 🔹 Gráficos lado a lado
col_g1, col_g2 = st.columns(2)

with col_g1:
    st.subheader("📊 Distribuição (Barras)")
    estado_counts = df["estado"].value_counts().reset_index()
    estado_counts.columns = ["estado", "Quantidade"]
    fig_bar = px.bar(estado_counts,
                     x="estado", y="Quantidade",
                     labels={"estado": "Estado", "Quantidade": "Quantidade"},
                     color="estado")
    st.plotly_chart(fig_bar, width='stretch')

with col_g2:
    st.subheader("🥧 Distribuição (Pizza)")
    fig_pie = px.pie(df, names="estado", title="Proporção de Transações")
    st.plotly_chart(fig_pie, width='stretch')
