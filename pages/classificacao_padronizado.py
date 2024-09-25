import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import plotly.figure_factory as ff
from utils.data_utils import read_df
from xgboost import XGBClassifier  # Importando o XGBoost

# Carregar o dataset
df_scaled = read_df('scaled_ACC_INTAKES_OUTCOMES', extension='parquet')

# 2. Configuração da Interface
st.write('<h1>Realize uma simulação com modelos de Classificação</h1>', unsafe_allow_html=True)
st.write('''<p>Para esta classificação, serão usados os algoritmos Random Forest, SVM e XGBoost. As características que serão usadas para 
         a seleção são referentes aos animais que passaram pelo Centro de Animal de Austin. A seleção destas colunas tem como 
         foco acertar as ocorrências de adoção, transferência, eutanásia, morto ou outro. Após a predicção, é exibido a porcentagem de acerto das amostras de treino e teste.</p>''', unsafe_allow_html=True)

st.write('----')

# Características para Seleção
features = {
    "Ano de nascimento": ["dob_year_scaled"],
    "Idade no momento de entrada": ["age_upon_intake_(years)_scaled"],
    "Ano de entrada": ["intake_year_scaled"],
    "Idade no momento de saída": ["age_upon_outcome_(years)_scaled"],
    "Ano de saída": ["outcome_year_scaled"],
    "Dias no abrigo": ["time_in_shelter_days_scaled"],
    "Sexo do animal na saída": ["sex_upon_outcome_Intact Female", "sex_upon_outcome_Intact Male  ", 
                                "sex_upon_outcome_Neutered Male", "sex_upon_outcome_Spayed Female", 
                                "sex_upon_outcome_Unknown      "],
    "Dia da semana na saída": ["outcome_weekday_Friday   ", "outcome_weekday_Monday   ", 
                               "outcome_weekday_Saturday ", "outcome_weekday_Sunday   ", 
                               "outcome_weekday_Thursday ", "outcome_weekday_Tuesday  ", 
                               "outcome_weekday_Wednesday"],
    "Tipo de animal": ["animal_type_Bir", "animal_type_Cat", "animal_type_Dog", "animal_type_Oth"],
    "Condição de saúde no momento da entrada": ["intake_condition_Aged   ", "intake_condition_Feral  ", 
                                                "intake_condition_Injured", "intake_condition_Normal ", 
                                                "intake_condition_Nursing", "intake_condition_Other  ", 
                                                "intake_condition_Pregnan", "intake_condition_Sick   "],
    "Tipo de entrada do animal": ["intake_type_Euthanasia Request", "intake_type_Owner Surrender   ", 
                                  "intake_type_Public Assist     ", "intake_type_Stray             ", 
                                  "intake_type_Wildlife          "],
    "Sexo do animal na entrada": ["sex_upon_intake_Intact Female", "sex_upon_intake_Intact Male  ", 
                                  "sex_upon_intake_Neutered Male", "sex_upon_intake_Spayed Female", 
                                  "sex_upon_intake_Unknown      "],
    "Dia da semana na entrada": ["intake_weekday_Friday   ", "intake_weekday_Monday   ", 
                                 "intake_weekday_Saturday ", "intake_weekday_Sunday   ", 
                                 "intake_weekday_Thursday ", "intake_weekday_Tuesday  ", 
                                 "intake_weekday_Wednesday"],
    "Raça Pura/Misturado": ["is_mix_breed"],
    "Cor": ["color_Black", "color_Brown/Chocolate", "color_Gray/Blue", "color_Other_Colors", 
            "color_Patterned", "color_Red/Orange", "color_White", "color_Yellow/Gold/Cream"]
}

selected_features = st.multiselect("Características", list(features.keys()), default=["Ano de nascimento", "Idade no momento de entrada"])

# Converter as seleções para as colunas reais do DataFrame
columns_selected = []
for feature in selected_features:
    columns_selected.extend(features[feature])

# 3. Processamento dos Dados
X = df_scaled[columns_selected]

# Criar a nova coluna "outcome_type_Others" com a operação bitwise OR para combinar "Missing" e "Return to Owner"
df_scaled['outcome_type_Others'] = df_scaled['outcome_type_Missing        '] | df_scaled['outcome_type_Return to Owner']

# Remover as colunas originais "Missing" e "Return to Owner"
df_scaled = df_scaled.drop(columns=['outcome_type_Missing        ', 'outcome_type_Return to Owner'])

# Mapeamento das colunas para labels legíveis
label_mapping = {
    "outcome_type_Adoption       ": "Adotado",
    "outcome_type_Euthanasia     ": "Eutanásia",
    "outcome_type_Died           ": "Morto",
    "outcome_type_Others": "Outro",
    "outcome_type_Transfer       ": "Transferido",
}

# Atualizar a variável de destino (y) para incluir as novas colunas-alvo
y = df_scaled[["outcome_type_Adoption       ", "outcome_type_Euthanasia     ", "outcome_type_Transfer       ", "outcome_type_Others", "outcome_type_Died           "]]

# Divisão dos dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Convertendo o DataFrame y_train para uma coluna única, com a classe dominante para cada instância
y_train_combined = y_train.idxmax(axis=1)

# Mapeando as classes categóricas para valores numéricos
class_mapping = {col: i for i, col in enumerate(y.columns)}
y_train_mapped = y_train_combined.map(class_mapping)
y_test_mapped = y_test.idxmax(axis=1).map(class_mapping)

# Balanceamento dos dados (SMOTE) no conjunto de treino
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train_mapped)

# 4. Treinamento do Modelo
algorithm = st.selectbox("Escolha o Algoritmo", ["Random Forest", "SVM", "XGBoost"])

if algorithm == "Random Forest":
    model = RandomForestClassifier(random_state=42)
elif algorithm == "SVM":
    model = SVC(random_state=42)
elif algorithm == "XGBoost":
    model = XGBClassifier(random_state=42)

model.fit(X_train_balanced, y_train_balanced)

# Previsões
y_test_pred_mapped = model.predict(X_test)

# Reverter os valores numéricos para os rótulos de classe originais
y_test_pred = pd.Series(y_test_pred_mapped).map({v: k for k, v in class_mapping.items()}).map(label_mapping)
y_test_mapped_labels = y_test_mapped.map({v: k for k, v in class_mapping.items()}).map(label_mapping)

# Cálculo das Acurácias
train_accuracy = accuracy_score(y_train_balanced, model.predict(X_train_balanced))
test_accuracy = accuracy_score(y_test_mapped, y_test_pred_mapped)

# Exibir as porcentagens de acerto
st.write(f"A porcentagem de acerto para o treino foi: <span style='color:red;'>{train_accuracy:.2%}</span>", unsafe_allow_html=True)
st.write(f"A porcentagem de acerto para o teste foi: <span style='color:red;'>{test_accuracy:.2%}</span>", unsafe_allow_html=True)

st.write('----')

st.write('<h2>Métricas de Classificação</h2>', unsafe_allow_html=True)
st.table(pd.DataFrame(classification_report(y_test_mapped_labels, y_test_pred, target_names=list(label_mapping.values()), output_dict=True)).T)

st.write('----')

st.write('<h2>Matriz de Confusão</h2>', unsafe_allow_html=True)
st.write('''É uma tabela que resume o desempenho de um modelo de classificação. 
                 Essa tabela fornece uma visão detalhada dos acertos e erros do modelo, sendo importante 
                 para avaliar sua eficácia e identificar áreas de melhoria.''')

conf_matrix = confusion_matrix(y_test_mapped_labels, y_test_pred)
conf_matrix_df = pd.DataFrame(
    conf_matrix,
    index=list(label_mapping.values()),
    columns=[f"Previsto: {label}" for label in list(label_mapping.values())]
)
st.table(conf_matrix_df)

# Exibir feature importance se o algoritmo for Random Forest ou XGBoost
if algorithm in ["Random Forest", "XGBoost"]:
    feature_importances = model.feature_importances_
    st.write('<h2>Feature Importances</h2>', unsafe_allow_html=True)
    st.write("A importância das características mostra quanto cada feature contribui para as decisões do modelo.")
    st.table(pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False))
