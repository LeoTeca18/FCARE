import streamlit as st
import pandas as pd

st.set_page_config(page_title="Transações Suspeitas - FCARE", page_icon="🚨", layout="wide")

st.title("🚨 Transações Suspeitas")

if "dataset" in st.session_state:
    df = st.session_state["dataset"]
else:
    st.warning("Nenhum dataset carregado. Volte à página inicial e carregue o arquivo.")
    st.stop()
# Filtrando apenas suspeitas (≥75%)
df_suspeitas = df[df["probabilidade_fraude"] >= 75]

st.subheader("🔍 Analisar Transações do Usuário")
id_selecionado = st.selectbox("Selecione o ID do usuário:", df_suspeitas["id"].unique())

if id_selecionado:
    transacoes_usuario = df_suspeitas[df_suspeitas["id"] == id_selecionado]
    st.write(f"**Transações suspeitas do usuário {transacoes_usuario.iloc[0]['nome']}:**")
    st.dataframe(transacoes_usuario, width='stretch')

    # Seleciona a transação pelo índice
    transacoes_indices = transacoes_usuario.index.tolist()
    transacao_escolhida = st.selectbox(
        "Selecione a transação para análise:",
        transacoes_indices,
        format_func=lambda idx: f"ID: {transacoes_usuario.loc[idx, 'id']} | Valor: {transacoes_usuario.loc[idx, 'valor_gasto_real']:.2f} | Hora: {transacoes_usuario.loc[idx, 'hora']}"
    )

    transacao = df.loc[transacao_escolhida]
    st.write("**Usuário:**", transacao["nome"])
    st.write("**Valor:**", f"{transacao['valor_gasto_real']:.2f}", "Kz")
    st.write("**Hora:**", transacao["hora"])
    st.write("**Localização:**", transacao["localizacao_desc"])
    st.write("**Tipo de Fraude:**", transacao["tipo_fraude"])
    st.write("**Probabilidade:**", f"{transacao['probabilidade_fraude']}%")

    acao = st.radio(
        "Classifique a transação:",
        ("✅ Confirmar Fraude", "❌ Rejeitar Suspeita"),
        horizontal=True
    )
    if st.button("Salvar classificação"):
        if acao == "✅ Confirmar Fraude":
            df.at[transacao_escolhida, "classe"] = 1
            df.at[transacao_escolhida, "estado"] = "Fraudulenta"
            st.success("Transação confirmada como fraude!")
        else:
            df.at[transacao_escolhida, "classe"] = 0
            df.at[transacao_escolhida, "estado"] = "Legítima"
            st.info("Suspeita rejeitada. Transação marcada como legítima.")
        # Atualiza o dataset na sessão
        st.session_state["dataset"] = df
