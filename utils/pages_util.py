import pandas as pd
import streamlit as st
from utils.data_utils import read_df
from st_pages import Page, show_pages, add_page_title


show_pages(
    [
        Page("Home.py", "Início", "🏠"),
        #Page("pages/exploratory_analisys.py", "Análises Exploratórias", "📝"),
        #Page("pages/preprocess.py", "Pré-processamento", "⚙️"),
        Page("pages/clusterizacao.py", "Clusterização", "🫧"),
        #("pages/colunas.py", "Colunas", "🫧"),
        Page("pages/classificacao_padronizado.py", "Simule com modelos de Classificação", "🗃️"), 
        Page("pages/grafico1.py", "EDA - Distribuição por Idade e Gênero", "📊"),
        Page("pages/grafico2.py", "EDA - Tipo de Entrada/Saída e Tipo de Animal", "📊"),
        Page("pages/grafico3.py", "EDA - Condição de Entrada e Tipo de Animal", "📊"),
        Page("pages/grafico4.py", "EDA - Radar de Entradas e Saídas por meses", "📊"),
    ]
)
def build_dataframe_section(df:pd.DataFrame):
    st.write('<h2>Dados do Centro de Animal de Austin</h2>', unsafe_allow_html=True)
    st.dataframe(df)


def read_ACC_df() -> pd.DataFrame:
    df = read_df('ACC_INTAKES_OUTCOMES', extension='parquet')
    return df

def dicionario_acc():
    with st.expander('Dicionário de dados do Centro Animal de Austin'):
        st.write(
        '''
        <table>
            <tr><th>COLUNA ORIGINAL</th><th>DESCRIÇÃO</th></tr>
            <tr><td>age_upon_outcome</td><td>Idade do animal na saída</td></tr>
            <tr><td>animal_id_outcome</td><td>ID de saída do animal. Deve corresponder à coluna de entrada</td></tr>
            <tr><td>date_of_birth</td><td>Data de nascimento do animal. Estimada se a data exata de nascimento não for conhecida</td></tr>
            <tr><td>outcome_subtype</td><td>Tipo de saída mais específico correspondente ao tipo de saída quando apropriado</td></tr>
            <tr><td>outcome_type</td><td>O tipo de saída</td></tr>
            <tr><td>sex_upon_outcome</td><td>O sexo do animal e se foi castrado ou esterilizado no momento da saída</td></tr>
            <tr><td>age_upon_outcome_(days)</td><td>A idade do animal na saída representada em dias</td></tr>
            <tr><td>age_upon_outcome_(years)</td><td>A idade do animal na saída representada em anos</td></tr>
            <tr><td>age_upon_outcome_age_group</td><td>Grupo de faixas etárias dos animais na saída. Vai por incrementos de 2.5 anos</td></tr>
            <tr><td>outcome_datetime</td><td>Data e hora da saída</td></tr>
            <tr><td>outcome_month</td><td>O mês representado como um valor numérico de 1-12 de quando a saída ocorreu</td></tr>
            <tr><td>outcome_year</td><td>O ano da saída</td></tr>
            <tr><td>outcome_monthyear</td><td>Mês e ano da saída representados como uma data e hora</td></tr>
            <tr><td>outcome_weekday</td><td>Dia da semana da saída</td></tr>
            <tr><td>outcome_hour</td><td>Hora da saída representada como um valor numérico de 1-24</td></tr>
            <tr><td>outcome_number</td><td>Valor numérico indicando se um animal foi liberado do abrigo mais de uma vez. Valores maiores que 1 indicam que o animal foi trazido e deixado no abrigo mais de uma vez</td></tr>
            <tr><td>dob_year</td><td>O ano de nascimento do animal</td></tr>
            <tr><td>dob_month</td><td>O mês de nascimento do animal como um mês numérico</td></tr>
            <tr><td>dob_monthyear</td><td>Data e hora do nascimento</td></tr>
            <tr><td>age_upon_intake</td><td>A idade do animal na entrada</td></tr>
            <tr><td>animal_id_intake</td><td>O ID único atribuído ao animal na entrada. Deve corresponder ao ID de saída do animal</td></tr>
            <tr><td>animal_type</td><td>Tipo de animal. Pode ser um dos 'gato', 'cachorro', 'pássaro', etc.</td></tr>
            <tr><td>breed</td><td>Raça do animal</td></tr>
            <tr><td>color</td><td>Cor do animal</td></tr>
            <tr><td>found_location</td><td>Endereço da rua ou área geral onde o animal foi encontrado</td></tr>
            <tr><td>intake_condition</td><td>A condição de entrada do animal. Pode ser uma das 'normal', 'ferido', 'doente', etc.</td></tr>
            <tr><td>intake_type</td><td>O tipo de entrada, por exemplo, 'abandonado', 'rendição de proprietário', etc.</td></tr>
            <tr><td>sex_upon_intake</td><td>O sexo do animal e se foi castrado ou esterilizado no momento da entrada</td></tr>
            <tr><td>count</td><td>Coluna auxiliar para tabular contagens. Todas as linhas nesta coluna são 1.</td></tr>
            <tr><td>age_upon_intake_(days)</td><td>A idade do animal na entrada representada em dias</td></tr>
            <tr><td>age_upon_intake_(years)</td><td>A idade do animal na entrada representada em anos</td></tr>
            <tr><td>age_upon_intake_age_group</td><td>Grupo de faixas etárias dos animais na entrada. Grupos são em incrementos de 2.5 anos</td></tr>
            <tr><td>intake_datetime</td><td>Data e hora da entrada</td></tr>
            <tr><td>intake_month</td><td>Mês numérico de quando a entrada ocorreu</td></tr>
            <tr><td>intake_year</td><td>Ano da entrada</td></tr>
            <tr><td>intake_monthyear</td><td>Mês e ano da entrada representados como uma data e hora</td></tr>
            <tr><td>intake_weekday</td><td>O dia da semana quando a entrada ocorreu</td></tr>
            <tr><td>intake_hour</td><td>Hora da entrada representada como um valor de 1 a 24</td></tr>
            <tr><td>intake_number</td><td>O número de entradas denotando o número de vezes que o animal foi trazido para o abrigo. Valores maiores que 1 indicam que o animal foi levado para o abrigo mais de uma vez</td></tr>
            <tr><td>time_in_shelter</td><td>O tempo no abrigo originalmente representado como um objeto TimeDelta</td></tr>
            <tr><td>time_in_shelter_days</td><td>Valor numérico indicando o número de dias que o animal permaneceu no abrigo da entrada à saída</td></tr>
        </table>
        <br>
''', unsafe_allow_html=True)