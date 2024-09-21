import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.data_utils import read_df

st.write('<h1>Entradas e Saídas por de acordo com os meses do ano</h1>', unsafe_allow_html=True)

# Carregar dataset
df = read_df('ACC_INTAKES_OUTCOMES')

# Seleção de anos múltiplos
selected_years = st.multiselect(
    "Selecione os anos para exibir no gráfico",
    options=df['intake_year'].unique(),
    default=[2013]
)

# Mapeamento de meses
month_map = {
    1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun',
    7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'
}

# Cartela de cores para cada ano
color_map = {
    2013: {'entrada': 'blue', 'saida': 'red'},
    2014: {'entrada': 'green', 'saida': 'orange'},
    2015: {'entrada': 'purple', 'saida': 'pink'},
    2016: {'entrada': 'cyan', 'saida': 'magenta'},
    2017: {'entrada': 'yellow', 'saida': 'brown'},
    2018: {'entrada': 'teal', 'saida': 'gold'}
}

# Processamento dos dados
df_radar = df[df['intake_year'].isin(selected_years)][['intake_year', 'outcome_month', 'intake_month']].copy()
df_radar['outcome_month'] = df_radar['outcome_month'].map(month_map)
df_radar['intake_month'] = df_radar['intake_month'].map(month_map)

# Agrupamento dos dados
df_radar_grouped = df_radar.groupby(['intake_year', 'intake_month', 'outcome_month']).size().reset_index(name='count')

# Preparação do gráfico radar
categories = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']

fig_radar = go.Figure()

# Variável para armazenar o valor máximo para definir o range adequado
max_value = 0

for year in selected_years:
    # Filtrar os dados para o ano
    df_year = df_radar_grouped[df_radar_grouped['intake_year'] == year]
    
    # Somar entradas e saídas por mês
    entradas = [df_year[df_year['intake_month'] == month]['count'].sum() for month in categories]
    saidas = [df_year[df_year['outcome_month'] == month]['count'].sum() for month in categories]
    
    # Encontrar o valor máximo para ajustar o range do gráfico
    max_value = max(max_value, max(entradas + saidas))
    
    # Adicionar a linha de Entrada ao gráfico com a cor associada ao ano
    fig_radar.add_trace(go.Scatterpolar(
        r=entradas,
        theta=categories,
        name=f'Entradas {year}',
        line=dict(color=color_map[year]['entrada'])
    ))
    
    # Adicionar a linha de Saída ao gráfico com a cor associada ao ano
    fig_radar.add_trace(go.Scatterpolar(
        r=saidas,
        theta=categories,
        name=f'Saídas {year}',
        line=dict(color=color_map[year]['saida'])
    ))

# Configurações do gráfico radar
fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(visible=True, range=[0, max_value * 1.1])  # range ajustado com base no valor máximo + 10% de folga
    ),
    showlegend=True,
    title="Entradas e Saídas por Mês (Radar Chart)"
)

# Exibir gráfico
st.plotly_chart(fig_radar)
