import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import plotly.figure_factory as ff
from utils.data_utils import read_df


# Carregar o dataset
df_scaled = read_df('scaled_ACC_INTAKES_OUTCOMES', extension='parquet')


# 2. Configuração da Interface
st.write('<h1>Classificação - Dataset Padronizado</h1>', unsafe_allow_html=True)
st.write('''<p>Para esta classificação, serão usados os algoritmos Random Forest e SVM. As características que serão usadas para 
         a seleção são referentes aos animais que passaram pelo Centro de Animal de Austin. A seleção destas colunas tem como 
         foco acertar as ocorrências de adoção, transferência ou eutanásia. Após a predicção, é exibido a porcentagem de acerto das amostras de treino e teste.</p>''', unsafe_allow_html=True)

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
y = df_scaled[["outcome_type_Adoption       ", "outcome_type_Euthanasia     ", "outcome_type_Transfer       "]]

# Divisão dos dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Balanceamento dos dados
smote = SMOTE(random_state=42)
y_train_combined = y_train.idxmax(axis=1)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train_combined)

# 4. Treinamento do Modelo
algorithm = st.selectbox("Escolha o Algoritmo", ["Random Forest", "SVM"])

if algorithm == "Random Forest":
    model = RandomForestClassifier(random_state=42)
elif algorithm == "SVM":
    model = SVC(random_state=42)

model.fit(X_train_balanced, y_train_balanced)

# Previsões
y_train_pred = model.predict(X_train_balanced)
y_test_pred = model.predict(X_test)

# Cálculo das Acurácias
train_accuracy = accuracy_score(y_train_balanced, y_train_pred)
test_accuracy = accuracy_score(y_test.idxmax(axis=1), y_test_pred)

# Exibir as porcentagens de acerto
st.write(f"A porcentagem de acerto para o treino foi: <span style='color:red;'>{train_accuracy:.2%}</span>", unsafe_allow_html=True)
st.write(f"A porcentagem de acerto para o teste foi: <span style='color:red;'>{test_accuracy:.2%}</span>", unsafe_allow_html=True)

st.write('----')

st.write('<h2>Métricas de Classificação</h2>', unsafe_allow_html=True)
st.table(pd.DataFrame(classification_report(y_test.idxmax(axis=1), y_test_pred, target_names=['Adoption', 'Euthanasia', 'Transfer'], output_dict=True)).T)

st.write('----')


st.write('<h2>Matriz de Confusão</h2>', unsafe_allow_html=True)
st.write('''É uma tabela que resume o desempenho de um modelo de classificação, destacando 
                 Verdadeiros Positivos (VP), Falsos Positivos (FP), Falsos Negativos (FN) e Verdadeiros Negativos (VN). 
                 Essa tabela fornece uma visão detalhada dos acertos e erros do modelo, sendo importante 
                 para avaliar sua eficácia e identificar áreas de melhoria.''')

conf_matrix = confusion_matrix(y_test.idxmax(axis=1), y_test_pred)
conf_matrix_df = pd.DataFrame(
    conf_matrix,
    index=['Adoption', 'Euthanasia', 'Transfer'],
    columns=['Previsto: Adoption', 'Previsto: Euthanasia', 'Previsto: Transfer']
)
st.table(conf_matrix_df)

fig_conf_matrix = ff.create_annotated_heatmap(
    z=conf_matrix,
    x=['Previsto: Adoption', 'Previsto: Euthanasia', 'Previsto: Transfer'],
    y=['Adoption', 'Euthanasia', 'Transfer'],
    colorscale='Blues',
    showscale=False
)
fig_conf_matrix.update_layout(
    title='Matriz de Confusão',
    xaxis_title='Previsto',
    yaxis_title='Real'
)
st.plotly_chart(fig_conf_matrix)

st.write('----')
# Se o modelo for Random Forest, mostrar feature importance
if algorithm == "Random Forest":
    feature_importances = model.feature_importances_
    st.write('<h2>Feature Importances</h2>', unsafe_allow_html=True)
    st.write("A importância das características mostra quanto cada feature contribui para as decisões do modelo.")
    st.table(pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False))
