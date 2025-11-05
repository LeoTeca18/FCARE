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

# 🔹 Gráfico dos tipos de fraude (apenas para transações marcadas como fraudulentas)
st.markdown("---")
st.subheader("🚨 Tipos de Fraude")
if "tipo_fraude" in df.columns:
    # Filtra apenas fraudes com tipo definido
    tipos = df.loc[(df["classe"] == 1) & (df["tipo_fraude"].notna()), "tipo_fraude"]
    if tipos.shape[0] > 0:
        tipos_counts = tipos.value_counts().reset_index()
        tipos_counts.columns = ["tipo_fraude", "Quantidade"]

        col_f1, col_f2 = st.columns([2, 1])
        with col_f1:
            st.subheader("Contagem por Tipo de Fraude")
            fig_tipos = px.bar(tipos_counts, x="tipo_fraude", y="Quantidade",
                               labels={"tipo_fraude": "Tipo de Fraude", "Quantidade": "Quantidade"},
                               color="tipo_fraude")
            st.plotly_chart(fig_tipos, width='stretch')

        with col_f2:
            st.subheader("Proporção por Tipo")
            fig_tipos_pie = px.pie(tipos_counts, names="tipo_fraude", values="Quantidade",
                                   title="Proporção dos Tipos de Fraude")
            st.plotly_chart(fig_tipos_pie, width='stretch')
    else:
        st.info("Não existem transações fraudulentas com 'tipo_fraude' definido no dataset.")
else:
    st.info("Coluna 'tipo_fraude' não encontrada no dataset. Carregue os dados pela página de upload para gerar essa coluna.")
