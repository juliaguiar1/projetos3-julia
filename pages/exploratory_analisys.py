import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_utils import read_df

st.write('<h1>Análises Explorátorias</h1>', unsafe_allow_html=True)


age_group_column = st.selectbox(
    "Escolha o grupo de idades para exibição:",
    ["Grupo de idade no momento de entrada", "Grupo de idade no momento de saída"]
)


if age_group_column == "Grupo de idade no momento de entrada":
    age_group_col = 'age_upon_intake_age_group'
else:
    age_group_col = 'age_upon_outcome_age_group'

df = read_df('ACC_INTAKES_OUTCOMES')

df_pyramid = df[['sex_upon_intake', age_group_col]].copy()
df_pyramid['sex_upon_intake'] = df_pyramid['sex_upon_intake'].apply(lambda x: 'Fêmea' if 'female' in x.lower() else ('Macho' if 'male' in x.lower() else 'Desconhecido'))
df_pyramid = df_pyramid[df_pyramid['sex_upon_intake'].isin(['Fêmea', 'Macho', 'Desconhecido'])]


age_group_order = [
    '(-0.025, 2.5',
    '(2.5, 5.0]  ',
    '(5.0, 7.5]  ',
    '(7.5, 10.0] ',
    '(10.0, 12.5]',
    '(12.5, 15.0]',
    '(15.0, 17.5]',
    '(17.5, 20.0]',
    '(20.0, 22.5]',
    '(22.5, 25.0]'
]

df_pyramid[age_group_col] = pd.Categorical(df_pyramid[age_group_col], categories=age_group_order, ordered=True)


df_pyramid_grouped = df_pyramid.groupby([age_group_col, 'sex_upon_intake'], observed=False).size().reset_index(name='count')


df_pyramid_grouped = df_pyramid_grouped.pivot(index=age_group_col, columns='sex_upon_intake', values='count').fillna(0)


df_pyramid_grouped['Macho'] = -df_pyramid_grouped['Macho']


fig = px.bar(df_pyramid_grouped,
             x=['Macho', 'Fêmea', 'Desconhecido'],
             y=df_pyramid_grouped.index,
             orientation='h',
             title=f'Pirâmide Populacional por {age_group_column} e Gênero',
             labels={'value': 'Contagem', age_group_col: 'Faixa Etária', 'variable': 'Gênero'},
             color_discrete_map={'Macho': 'lightblue', 'Fêmea': 'lightpink', 'Desconhecido': 'lightgray'})

fig.update_layout(barmode='overlay')


st.plotly_chart(fig)
st.write('----')


df_bubble = df[['sex_upon_outcome', 'animal_type', 'outcome_type']].copy()


df_bubble['sex_upon_outcome'] = df_bubble['sex_upon_outcome'].apply(
    lambda x: 'Fêmea' if 'female' in x.lower() else ('Macho' if 'male' in x.lower() else 'Desconhecido')
)


df_bubble['animal_type'] = df_bubble['animal_type'].replace({
    'Bir': 'Pássaro',
    'Cat': 'Gato',
    'Dog': 'Cachorro',
    'Oth': 'Outros'
})


df_bubble['outcome_type'] = df_bubble['outcome_type'].replace({
    'Adoption       ': 'Adoção',
    'Died           ': 'Morto',
    'Euthanasia     ': 'Eutanásia',
    'Missing        ': 'Desaparecido',
    'Return to Owner': 'Devolvido ao tutor(a)',
    'Transfer       ': 'Transferido'
})


df_bubble_grouped = df_bubble.groupby(['outcome_type', 'sex_upon_outcome', 'animal_type']).size().reset_index(name='count')


fig_bubble = px.scatter(df_bubble_grouped,
                        x='outcome_type',
                        y='sex_upon_outcome',
                        size='count',
                        color='animal_type',
                        title='Distribuição por Tipo de Saída, Gênero e Tipo de Animal',
                        labels={'sex_upon_outcome': 'Gênero', 'outcome_type': 'Tipo de Saída', 'animal_type': 'Tipo de Animal'},
                        hover_name='animal_type',
                        size_max=60,
                        color_discrete_map={
                            'Cachorro': 'red',
                            'Gato': 'blue',
                            'Pássaro': 'green',
                            'Outros': 'orange'
                        })

fig_bubble.update_layout(
    xaxis=dict(title='Tipo de Saída'),
    yaxis=dict(title='Gênero')
)

# Exibir o gráfico de bolhas 
st.plotly_chart(fig_bubble)

st.write('----')

df_bar = df[['animal_type', 'intake_condition']].copy()


df_bar['animal_type'] = df_bar['animal_type'].replace({
    'Bir': 'Pássaro',
    'Cat': 'Gato',
    'Dog': 'Cachorro',
    'Oth': 'Outros'
})


df_bar['intake_condition'] = df_bar['intake_condition'].replace({
    'Aged   ': 'Idoso',
    'Feral  ': 'Feroz',
    'Injured': 'Machucado',
    'Normal ': 'Normal',
    'Nursing': 'Amamentando',
    'Other  ': 'Outros',
    'Pregnan': 'Grávida',
    'Sick   ': 'Doente'
})


df_bar_grouped = df_bar.groupby(['animal_type', 'intake_condition']).size().reset_index(name='count')


fig_bar = px.bar(df_bar_grouped,
                 x='animal_type',
                 y='count',
                 color='intake_condition',
                 title='Distribuição de Condições de Entrada por Tipo de Animal',
                 labels={'animal_type': 'Tipo de Animal', 'intake_condition': 'Condição de Entrada', 'count': 'Contagem'},
                 color_discrete_map={
                     'Idoso': 'red',
                     'Feroz': 'blue',
                     'Machucado': 'green',
                     'Normal': 'purple',
                     'Amamentando': 'orange',
                     'Outros': 'pink',
                     'Grávida': 'yellow',
                     'Doente': 'brown'
                 })


fig_bar.update_layout(
    xaxis=dict(title='Tipo de Animal'),
    yaxis=dict(title='Contagem')
)


st.plotly_chart(fig_bar)

st.write('----')

selected_year = st.slider(
    "Selecione o ano para iteragir com o gráfico abaixo",
    min_value=2013,
    max_value=2018,
    value=2013
)

df_rose = df[df['intake_year'] == selected_year][['outcome_month', 'intake_month']].copy()

month_map = {
    1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun',
    7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'
}
df_rose['outcome_month'] = df_rose['outcome_month'].map(month_map)
df_rose['intake_month'] = df_rose['intake_month'].map(month_map)

df_rose_grouped = df_rose.melt(var_name='Movimentação', value_name='Month')

df_rose_grouped['Movimentação'] = df_rose_grouped['Movimentação'].replace({
    'intake_month': 'Entrada',
    'outcome_month': 'Saída'
})

df_rose_grouped = df_rose_grouped.groupby(['Movimentação', 'Month']).size().reset_index(name='count')

fig_rose = px.bar_polar(df_rose_grouped, r='count', theta='Month', color='Movimentação',
                        title=f'Entradas e Saídas por Mês no Ano {selected_year}',
                        color_discrete_map={'Entrada': 'lightblue', 'Saída': 'lightpink'},
                        category_orders={'Month': ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']})

fig_rose.update_layout(
    polar=dict(radialaxis=dict(visible=True)),
    showlegend=True
)

# Exibir o gráfico Nightingale Rose Chart
st.plotly_chart(fig_rose)
