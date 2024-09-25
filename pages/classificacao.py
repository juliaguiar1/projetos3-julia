import streamlit as st
import pandas as pd

# Dados do relatório de classificação para Random Forest
random_forest_report = {
    'Precisão': [0.85, 0.29, 0.83, 0.50, 0.89],
    'Recall': [0.85, 0.14, 0.74, 0.12, 0.93],
    'F1-Score': [0.85, 0.19, 0.78, 0.20, 0.91],
    'Support': [1742, 173, 1772, 8, 6614]
}

# Dados do relatório de classificação para SVM
svm_report = {
    'Precisão': [0.79, 0.10, 0.74, 0.33, 0.89],
    'Recall': [0.87, 0.20, 0.73, 0.12, 0.85],
    'F1-Score': [0.83, 0.13, 0.74, 0.18, 0.87],
    'Support': [1742, 173, 1772, 8, 6614]
}

# Dados do relatório de classificação para XGBoost
xgboost_report = {
    'Precisão': [0.86, 0.33, 0.87, 0.25, 0.88],
    'Recall': [0.84, 0.13, 0.73, 0.12, 0.94],
    'F1-Score': [0.85, 0.18, 0.79, 0.17, 0.91],
    'Support': [1742, 173, 1772, 8, 6614]
}

# Dados de Validação Cruzada
validation_data = {
    "Random Forest": {
        "Acurácia Fold 1": 0.92248866,
        "Acurácia Fold 2": 0.95994815,
        "Acurácia Fold 3": 0.96966948,
        "Acurácia Fold 4": 0.97148412,
        "Acurácia Fold 5": 0.97407647,
        "Média da Acurácia": 0.9595,
        "Desvio Padrão": 0.0191
    },
    "SVM": {
        "Acurácia Fold 1": 0.79157485,
        "Acurácia Fold 2": 0.82559948,
        "Acurácia Fold 3": 0.82838626,
        "Acurácia Fold 4": 0.82845107,
        "Acurácia Fold 5": 0.83596889,
        "Média da Acurácia": 0.8220,
        "Desvio Padrão": 0.0156
    },
    "XGBoost": {
        "Acurácia Fold 1": 0.87990927,
        "Acurácia Fold 2": 0.95191186,
        "Acurácia Fold 3": 0.96824368,
        "Acurácia Fold 4": 0.97012314,
        "Acurácia Fold 5": 0.97051199,
        "Média da Acurácia": 0.9481,
        "Desvio Padrão": 0.0348
    }
}

# Matrizes de Confusão corrigidas e colunas ajustadas
confusion_matrices = {
    "Random Forest": pd.DataFrame(
        [
            [1482, 9, 6, 0, 245],
            [8, 24, 46, 0, 94],
            [30, 19, 1304, 0, 419],
            [2, 0, 0, 1, 5],
            [223, 32, 221, 1, 6137]
        ],
        columns=['Adotado', 'Morto', 'Eutanasiado', 'Outro', 'Transferido'],
        index=['Adotado', 'Morto', 'Eutanasiado', 'Outro', 'Transferido']
    ),
    "SVM": pd.DataFrame(
        [
            [1508, 19, 196, 18, 1],
            [25, 1315, 396, 36, 0],
            [360, 383, 5619, 250, 2],
            [5, 57, 75, 34, 1],
            [1, 0, 4, 1, 2]
        ],
        columns=['Adotado', 'Eutanasiado', 'Transferido', 'Morto', 'Outro'],
        index=['Adotado', 'Eutanasiado', 'Transferido', 'Morto', 'Outro']
    ),
    "XGBoost": pd.DataFrame(
        [
            [1470, 3, 266, 3, 0],
            [23, 1295, 439, 15, 0],
            [215, 157, 6212, 27, 3],
            [8, 42, 100, 22, 0],
            [0, 0, 7, 0, 1]
        ],
        columns=['Adotado', 'Eutanasiado', 'Transferido', 'Morto', 'Outro'],
        index=['Adotado', 'Eutanasiado', 'Transferido', 'Morto', 'Outro']
    )
}

# Relatórios de Classificação
classification_reports = {
    "Random Forest": pd.DataFrame(random_forest_report, index=['Adotado', 'Morto', 'Eutanasiado', 'Outro', 'Transferido']),
    "SVM": pd.DataFrame(svm_report, index=['Adotado', 'Morto', 'Eutanasiado', 'Outro', 'Transferido']),
    "XGBoost": pd.DataFrame(xgboost_report, index=['Adotado', 'Morto', 'Eutanasiado', 'Outro', 'Transferido'])
}

# Seleção de modelo com dropdown
st.title("Seleção de Modelo - Random Forest, SVM e XGBoost")

model_selected = st.selectbox("Escolha o modelo", ["Random Forest", "SVM", "XGBoost"])

# Exibir informações com base na seleção do modelo
st.header(f"Relatório de Classificação - {model_selected}")
st.dataframe(classification_reports[model_selected])

st.header(f"Matriz de Confusão - {model_selected}")
st.dataframe(confusion_matrices[model_selected])

st.header(f"Validação Cruzada - {model_selected}")
validation_df = pd.DataFrame({
    'Métrica': ['Acurácia Fold 1', 'Acurácia Fold 2', 'Acurácia Fold 3', 'Acurácia Fold 4', 'Acurácia Fold 5', 'Média da Acurácia', 'Desvio Padrão'],
    'Valores': [
        validation_data[model_selected]['Acurácia Fold 1'],
        validation_data[model_selected]['Acurácia Fold 2'],
        validation_data[model_selected]['Acurácia Fold 3'],
        validation_data[model_selected]['Acurácia Fold 4'],
        validation_data[model_selected]['Acurácia Fold 5'],
        validation_data[model_selected]['Média da Acurácia'],
        validation_data[model_selected]['Desvio Padrão']
    ]
})
st.dataframe(validation_df)
