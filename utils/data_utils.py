import os
import pandas as pd

def df_names():
    result = []
    dir_iter = os.scandir('data')
    for f in dir_iter:
        if f.name.endswith('.csv'):
            result.append(f.name[0:-4])
    return sorted(result)

def read_df(df_name, extension='parquet', encoding='utf-8', low_memory=False):
    data_dir = 'data'  # Diretório onde o CSV está localizado
    parquet_path = os.path.join(data_dir, f'{df_name}.parquet')
    csv_path = os.path.join(data_dir, f'{df_name}.csv')
    
    # Verifica se o arquivo Parquet existe
    if not os.path.exists(parquet_path):
        # Se o arquivo Parquet não existir, converte o CSV para Parquet
        if os.path.exists(csv_path):
            df = __read_csv(csv_path, encoding=encoding, low_memory=low_memory)
            df.to_parquet(parquet_path, engine='pyarrow')
        else:
            raise FileNotFoundError(f"Arquivo CSV '{csv_path}' não encontrado.")
    
    # Lê o arquivo Parquet
    return pd.read_parquet(parquet_path)

def __read_csv(path, encoding, low_memory=False):
    try:
        df = pd.read_csv(path, sep=',', encoding=encoding, low_memory=low_memory)
    except Exception as e:
        print(f"Erro ao ler CSV: {e}. Tentando com separador ';'.")
        df = pd.read_csv(path, sep=';', encoding=encoding, low_memory=low_memory)
    return df
