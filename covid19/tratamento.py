import pandas as pd
import numpy as np

def load_and_prepare_data():

    # Caminho para o arquivo CSV
    file_path = 'datasets/World_covid_cases_deaths_vaccination_data.csv'

    # Carregar o dataset
    data = pd.read_csv(file_path)

    # Filtrando os dados para o Brasil
    data = data[data['country'] == 'brazil'].copy()

    # Converte a data para datetime e transforma em índice
    data.loc[:, 'date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
    data = data.set_index('date')

    # Defino a frequência diária.
    data = data.asfreq('D')

    start_date = pd.to_datetime('2020-06-02') # Primeiro caso confirmado no Brasil

    # Tratar valores negativos como NaN
    data['daily_cases'] = data['daily_cases'].apply(lambda x: x if x >= 0 else None)
    data['total_vaccinations'] = data['total_vaccinations'].apply(lambda x: x if x >= 0 else None)

    # Tratar valores 0 em daily_cases após o início dos casos
    data.loc[(data.index >= start_date) & (data['daily_cases'] == 0), 'daily_cases'] = None


    # Preenchendo valores ausentes por interpolação
    data['daily_cases'] = data['daily_cases'].interpolate(method='linear')
    data['total_vaccinations'] = data['total_vaccinations'].interpolate(method='linear')

    # Ordena pela data
    data = data.sort_index()

    print(f'Dados carregados')
    return data
