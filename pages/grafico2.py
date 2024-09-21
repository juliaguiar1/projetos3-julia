import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_utils import read_df

st.write('<h1>Distribuição por Tipo de Entrada/Saída e Tipo de Animal</h1>', unsafe_allow_html=True)

# Carregar dataset
df = read_df('ACC_INTAKES_OUTCOMES')

# Modificar valores para exibição mais amigável
df['animal_type'] = df['animal_type'].replace({
    'Bir': 'Pássaro', 'Cat': 'Gato', 'Dog': 'Cachorro', 'Oth': 'Outros'
})
df['outcome_type'] = df['outcome_type'].replace({
    'Adoption       ': 'Adoção', 'Died           ': 'Morto', 'Euthanasia     ': 'Eutanásia',
    'Missing        ': 'Desaparecido', 'Return to Owner': 'Devolvido ao tutor(a)', 'Transfer       ': 'Transferido'
})
df['intake_type'] = df['intake_type'].replace({
    'Euthanasia Request': 'Pedido de Eutanásia', 'Owner Surrender   ': 'Entrega Voluntária',
    'Public Assist     ': 'Assistência Pública', 'Stray             ': 'Animal de Rua', 'Wildlife          ': 'Vida Selvagem'
})

# Dropdown para escolher exibição por tipo de entrada ou saída
option = st.selectbox(
    'Escolha o tipo de exibição:',
    ['Tipo de Saída', 'Tipo de Entrada']
)

# Dropdown para filtrar por tipo de animal
animal_filter = st.selectbox(
    'Escolha o tipo de animal para filtrar:',
    ['Todos', 'Cachorro', 'Gato', 'Pássaro', 'Outros']
)

# Condicional para ajustar o agrupamento com base na escolha
if option == 'Tipo de Saída':
    df_grouped = df.groupby(['outcome_type', 'animal_type']).size().reset_index(name='count')
    x_axis = 'outcome_type'
    x_title = 'Tipo de Saída'
else:
    df_grouped = df.groupby(['intake_type', 'animal_type']).size().reset_index(name='count')
    x_axis = 'intake_type'
    x_title = 'Tipo de Entrada'

# Aplicar filtro de tipo de animal, se selecionado
if animal_filter != 'Todos':
    df_grouped = df_grouped[df_grouped['animal_type'] == animal_filter]

# Criar gráfico de barras
fig_bar = px.bar(df_grouped,
                 x=x_axis,
                 y='count',
                 color='animal_type',
                 barmode='group',
                 title=f'Distribuição por {x_title} e Tipo de Animal',
                 labels={'count': 'Contagem', x_axis: x_title, 'animal_type': 'Tipo de Animal'},
                 hover_name='animal_type',
                 color_discrete_map={
                     'Cachorro': 'red', 'Gato': 'blue', 'Pássaro': 'green', 'Outros': 'orange'
                 })

fig_bar.update_layout(
    xaxis=dict(title=x_title),
    yaxis=dict(title='Contagem'),
    legend_title='Tipo de Animal'
)

# Exibir gráfico
st.plotly_chart(fig_bar)
