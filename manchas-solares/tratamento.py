import pandas as pd
import numpy as np

def load_and_prepare_data():

    # Caminho para o arquivo CSV
    file_path = 'datasets/Sunspots.csv'

    # Carregar o dataset
    data = pd.read_csv(file_path)

    # Renomear colunas
    data = data.rename(columns={ 'Date': 'date', 'Monthly Mean Total Sunspot Number': 'sunspots'})

    # Converte a data para datetime e transforma em índice
    data.loc[:, 'date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
    data = data.set_index('date')

    # Defino a frequência mensal.
    data = data.asfreq('ME')

    # Preenchendo valores ausentes por interpolação
    data['sunspots'] = data['sunspots'].interpolate(method='linear')
    
    # Ordena pela data
    data = data.sort_index()

    # Filtra os dados a partir de 1900
    start_date = pd.to_datetime('1900-01-01')
    data = data[data.index >= start_date]

    print(f'Dados carregados')
    return data
