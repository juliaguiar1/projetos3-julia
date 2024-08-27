import streamlit as st
import pandas as pd
from utils.data_utils import read_df

st.set_page_config(page_title="Visualizar Colunas dos Datasets", layout="wide")

def main():
    st.title("Visualização de Colunas dos Datasets")

    # Carregar os dados normalizados e padronizados
    normalized_data_name = 'normalized_ACC_INTAKES_OUTCOMES'
    scaled_data_name = 'scaled_ACC_INTAKES_OUTCOMES'
    
    data_normalized = read_df(normalized_data_name, extension='parquet')
    data_scaled = read_df(scaled_data_name, extension='parquet')

    # Dropdown para selecionar qual dataset visualizar
    dataset_option = st.selectbox('Selecione o dataset para visualizar as colunas:',
                                  ['Dataset Padronizado', 'Dataset Normalizado'])

    if dataset_option == 'Dataset Padronizado':
        st.write(f"**Colunas do Dataset Padronizado ({scaled_data_name}):**")
        st.write(data_scaled.columns.tolist())
    else:
        st.write(f"**Colunas do Dataset Normalizado ({normalized_data_name}):**")
        st.write(data_normalized.columns.tolist())

if __name__ == "__main__":
    main()
