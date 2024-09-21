import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_utils import read_df

st.write('<h1>Distribuição populacional por Idade e Gênero</h1>', unsafe_allow_html=True)

# Carregar dataset
df = read_df('ACC_INTAKES_OUTCOMES')

age_group_column = st.selectbox(
    "Escolha o grupo de idades para exibição:",
    ["Grupo de idade no momento de entrada", "Grupo de idade no momento de saída"]
)

if age_group_column == "Grupo de idade no momento de entrada":
    age_group_col = 'age_upon_intake_age_group'
else:
    age_group_col = 'age_upon_outcome_age_group'

df_pyramid = df[['sex_upon_intake', age_group_col]].copy()
df_pyramid['sex_upon_intake'] = df_pyramid['sex_upon_intake'].apply(
    lambda x: 'Fêmea' if 'female' in x.lower() else ('Macho' if 'male' in x.lower() else 'Desconhecido')
)
df_pyramid = df_pyramid[df_pyramid['sex_upon_intake'].isin(['Fêmea', 'Macho', 'Desconhecido'])]

age_group_order = [
    '(-0.025, 2.5', '(2.5, 5.0]  ', '(5.0, 7.5]  ', '(7.5, 10.0] ', '(10.0, 12.5]',
    '(12.5, 15.0]', '(15.0, 17.5]', '(17.5, 20.0]', '(20.0, 22.5]', '(22.5, 25.0]'
]

# Mapeamento dos rótulos para exibição no gráfico
age_group_labels = {
    '(-0.025, 2.5': '0 - 2.5',
    '(2.5, 5.0]  ': '2.5 - 5.0',
    '(5.0, 7.5]  ': '5.0 - 7.5',
    '(7.5, 10.0] ': '7.5 - 10.0',
    '(10.0, 12.5]': '10.0 - 12.5',
    '(12.5, 15.0]': '12.5 - 15.0',
    '(15.0, 17.5]': '15.0 - 17.5',
    '(17.5, 20.0]': '17.5 - 20.0',
    '(20.0, 22.5]': '20.0 - 22.5',
    '(22.5, 25.0]': '22.5 - 25.0'
}

df_pyramid[age_group_col] = pd.Categorical(df_pyramid[age_group_col], categories=age_group_order, ordered=True)

df_pyramid_grouped = df_pyramid.groupby([age_group_col, 'sex_upon_intake'], observed=False).size().reset_index(name='count')

# Substituir os rótulos no DataFrame para exibição correta no gráfico
df_pyramid_grouped[age_group_col] = df_pyramid_grouped[age_group_col].replace(age_group_labels)

# Dropdown para selecionar o gênero
selected_gender = st.selectbox(
    "Selecione o gênero para exibição:",
    ['Todos', 'Macho', 'Fêmea', 'Desconhecido']
)

# Filtrar dados com base no gênero selecionado
if selected_gender != 'Todos':
    df_pyramid_grouped = df_pyramid_grouped[df_pyramid_grouped['sex_upon_intake'] == selected_gender]

# Gráfico de barras filtrado
fig = px.bar(df_pyramid_grouped,
             x=age_group_col,
             y='count',
             color='sex_upon_intake',
             title=f'Distribuição Populacional por {age_group_column} e Gênero',
             labels={'count': 'Contagem', age_group_col: 'Faixa Etária', 'sex_upon_intake': 'Gênero'},
             color_discrete_map={'Macho': 'lightblue', 'Fêmea': 'lightpink', 'Desconhecido': 'lightgray'})

fig.update_layout(barmode='group', xaxis_title="Faixa Etária", yaxis_title="Contagem", legend_title="Gênero")

st.plotly_chart(fig)
