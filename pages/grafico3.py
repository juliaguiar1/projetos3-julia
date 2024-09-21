import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_utils import read_df

st.write('<h1>Distribuição de Condições de Entrada por Tipo de Animal</h1>', unsafe_allow_html=True)

# Carregar dataset
df = read_df('ACC_INTAKES_OUTCOMES')

df_bar = df[['animal_type', 'intake_condition']].copy()
df_bar['animal_type'] = df_bar['animal_type'].replace({
    'Bir': 'Pássaro', 'Cat': 'Gato', 'Dog': 'Cachorro', 'Oth': 'Outros'
})
df_bar['intake_condition'] = df_bar['intake_condition'].replace({
    'Aged   ': 'Idoso', 'Feral  ': 'Feroz', 'Injured': 'Machucado', 'Normal ': 'Normal',
    'Nursing': 'Amamentando', 'Other  ': 'Outros', 'Pregnan': 'Grávida', 'Sick   ': 'Doente'
})
df_bar_grouped = df_bar.groupby(['animal_type', 'intake_condition']).size().reset_index(name='count')

fig_bar = px.bar(df_bar_grouped,
                 x='animal_type',
                 y='count',
                 color='intake_condition',
                 title='Distribuição de Condições de Entrada por Tipo de Animal',
                 labels={'animal_type': 'Tipo de Animal', 'intake_condition': 'Condição de Entrada', 'count': 'Contagem'},
                 color_discrete_map={
                     'Idoso': 'red', 'Feroz': 'blue', 'Machucado': 'green', 'Normal': 'purple',
                     'Amamentando': 'orange', 'Outros': 'pink', 'Grávida': 'yellow', 'Doente': 'brown'
                 })
fig_bar.update_layout(
    xaxis=dict(title='Tipo de Animal'),
    yaxis=dict(title='Contagem')
)
st.plotly_chart(fig_bar)
