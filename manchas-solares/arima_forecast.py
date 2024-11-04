from tratamento import load_and_prepare_data

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import root_mean_squared_error, mean_absolute_error

from pmdarima.arima import auto_arima


# Configurações para os gráficos
plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['figure.figsize'] = (15, 6)

# Carrega os dados
data = load_and_prepare_data()

# Plota os dados
fig, ax1 = plt.subplots()

# Plota as manchas solares no gráfico
ax1.plot(data.index, data['sunspots'], color='tab:blue', label='Média de manchas solares mensais')

# Título
fig.suptitle('Média de manchas solares mensais')
plt.show()

# -----------------------------------------------------

# Dividir os dados em treino e teste
steps = 132 # 11 anos
data_train = data[:-steps]
data_test  = data[-steps:]

# Modelo ARIMA

model = auto_arima (
                    data_train['sunspots'],
                    trace=True, 
                    error_action='warn', 
                    suppress_warnings=True,
                   )

# Treina o modelo
model.fit(data_train['sunspots'])

# Previsao
forecast = model.predict(n_periods=len(data_test))

# Converte para um dataframe
forecast = pd.DataFrame(forecast,index = data_test.index,columns=['Prediction'])

# Métricas de avaliação do modelo
y_true = data_test['sunspots']
y_pred = forecast['Prediction']

# RMSE
rmse = root_mean_squared_error(y_true, y_pred)
print('\nMétricas de avaliação do modelo')
print(f'RMSE: {rmse}')

# MAE
mae = mean_absolute_error(y_true, y_pred)
print(f'MAE: {mae}')

# Plota a predição, comparando com valores reais
fig, ax=plt.subplots()
data_train['sunspots'].plot(ax=ax, label='train')
data_test['sunspots'].plot(ax=ax, label='test')
forecast['Prediction'].plot(ax=ax, label='predictions', color='green')
ax.legend()
plt.show()
