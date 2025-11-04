import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuração da página
st.set_page_config(page_title="Histórico do Usuário", page_icon="📜", layout="wide")

# Título
st.title("📜 Histórico do Usuário")

if "dataset" in st.session_state:
    df = st.session_state["dataset"]
else:
    st.warning("Nenhum dataset carregado. Volte à página inicial e carregue o arquivo.")
    st.stop()
    
# Escolher o ID do usuário
id_usuario = st.selectbox("🔎 Selecione o ID do Usuário", df["id"].unique())

# Buscar o nome correspondente ao ID
nome_usuario = df.loc[df["id"] == id_usuario, "nome"].iloc[0]

# Exibir tabela das transações do usuário selecionado
st.subheader(f"📊 Transações Passadas de {nome_usuario}")
historico_usuario = df[df["id"] == id_usuario].sort_values("hora", ascending=False)
st.dataframe(historico_usuario, width='stretch')

# Cálculo das métricas do usuário
media_gastos_usuario = historico_usuario["valor_gasto_real"].mean()
freq_transacoes_usuario = (
    len(historico_usuario) / ((historico_usuario["hora"].max() - historico_usuario["hora"].min()))
    if len(historico_usuario) > 1 else len(historico_usuario)
)

col1, col2 = st.columns(2)
with col1:
    st.metric("💰 Média de Gastos (Usuário)", f"{media_gastos_usuario:.2f} Kz")
with col2:
    st.metric("📅 Frequência Média (Usuário)", f"{freq_transacoes_usuario:.1f} transações")

# Gráfico de tendência do usuário
st.subheader(f"📈 Tendência de Gastos de {nome_usuario}")
df_grouped_usuario = historico_usuario.groupby("hora")["valor_gasto_real"].sum()

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df_grouped_usuario.index, df_grouped_usuario.values, marker='o')
ax.set_title(f"Tendência de Gastos de {nome_usuario} ao Longo do Tempo")
ax.set_xlabel("Hora")
ax.set_ylabel("Valor Total (Kz)")
ax.grid(True)
st.pyplot(fig)