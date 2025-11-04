import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Tela de Upload", page_icon="📊", layout="wide")

st.title("📊 Tela de Upload")

# Upload
arquivo = st.file_uploader("Selecione um CSV", type=["csv"])

if arquivo is not None:
    df = pd.read_csv(arquivo)

    st.subheader("Prévia do Dataset")
    st.dataframe(df.head())

    # 🔹 Carregar modelo treinado
    modelo = joblib.load("models/modelo_fraude_rf.pkl")

    # 🔹 Carregar scaler e encoders salvos
    scaler = joblib.load("models/scaler_valor_gasto.pkl")
    encoder_categoria = joblib.load("models/encoder_categoria_compra.pkl")
    encoder_tipo = joblib.load("models/encoder_tipo_transacao.pkl")
    encoder_localizacao = joblib.load("models/encoder_localizacao.pkl")
    encoder_banco = joblib.load("models/encoder_banco_emissor.pkl")
    encoder_faixa = joblib.load("models/encoder_faixa_horaria.pkl")
    encoder_online = joblib.load("models/encoder_online_x_faixa.pkl")

    # 🔹 Features que o modelo espera
    expected = list(modelo.feature_names_in_)

    # 🔹 Colunas a remover (id, nome, classe normalmente não são features)
    colunas_remover = ["id", "nome", "classe"]
    df_features = df.drop(columns=[c for c in colunas_remover if c in df.columns])

    # 🔹 Ajustar DataFrame às features esperadas
    missing = [c for c in expected if c not in df_features.columns]
    for col in missing:
        df_features[col] = 0   # preenche valores ausentes

    X = df_features.reindex(columns=expected)

    try:
        # 🔹 Predição de probabilidade
        probs = modelo.predict_proba(X)[:, 1] * 100  # Probabilidade de fraude (classe 1)

        # 🔹 Predição da classe (0 ou 1)
        predicoes = modelo.predict(X)

        # 🔹 Adicionar colunas ao dataset original
        df["classe"] = predicoes  # 0 = não fraude, 1 = fraude
        df["probabilidade_fraude"] = probs.round(2)  # Percentual de 0 a 100
                
        # 🔹 Desnormalizar valor gasto
        if "valor_gasto" in df_features.columns:
            df["valor_gasto_real"] = scaler.inverse_transform(df_features[["valor_gasto"]])

        # 🔹 Descodificar variáveis categóricas
        if "categoria_compra" in df_features.columns:
            df["categoria_compra_desc"] = encoder_categoria.inverse_transform(df_features["categoria_compra"])
        if "tipo_transacao" in df_features.columns:
            df["tipo_transacao_desc"] = encoder_tipo.inverse_transform(df_features["tipo_transacao"])
        if "localizacao" in df_features.columns:
            df["localizacao_desc"] = encoder_localizacao.inverse_transform(df_features["localizacao"])
        if "banco_emissor" in df_features.columns:
            df["banco_emissor_desc"] = encoder_banco.inverse_transform(df_features["banco_emissor"])
        if "faixa_horaria" in df_features.columns:
            df["faixa_horaria_desc"] = encoder_faixa.inverse_transform(df_features["faixa_horaria"])
        if "online_x_faixa" in df_features.columns:
            df["online_x_faixa_desc"] = encoder_online.inverse_transform(df_features["online_x_faixa"])

            
        st.success("Modelo adicionado ao dataset ✅")
        st.success("Colunas adicionadas ao dataset ✅")
        st.dataframe(df.head())

        # salvar no session_state
        st.session_state["dataset"] = df

    except Exception as e:
        st.error(f"Erro ao calcular probabilidades: {e}")
