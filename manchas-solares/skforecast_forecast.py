from tratamento import load_and_prepare_data

import matplotlib.pyplot as plt

import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import random_search_forecaster

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

# Hyperparameter Tunning
# print('Hyperparameter Tunning...')
# forecaster = ForecasterAutoreg(
#                  regressor = RandomForestRegressor(random_state=123),
#                  lags      = 10 
#              )

# # # Valores de lag a testar
# lags_grid = [50, 65]

# # Regressor hyperparameters
# param_distributions = {
#                 'n_estimators': np.arange(start=10, stop=500, step=1, dtype=int),
#                 'max_depth': np.arange(start=5, stop=100, step=1, dtype=int)
#             }

# results = random_search_forecaster(
#               forecaster           = forecaster,
#               y                    = data_train['sunspots'],
#               steps                = steps,
#               lags_grid            = lags_grid,
#               param_distributions  = param_distributions,
#               n_iter               = 20,
#               metric               = 'mean_squared_error',
#               refit                = False,
#               initial_train_size   = len(data_train) // 2,
#               fixed_train_size     = False,
#               return_best          = True,
#               random_state         = 123,
#               n_jobs               = 'auto',
#               verbose              = False,
#               show_progress        = True,
#           )

# Modelo selecionado
print('Treinando modelo...')
forecaster = ForecasterAutoreg(
                regressor = RandomForestRegressor(random_state=123, n_estimators=15, max_depth=47),
                lags      = 65
            )

forecaster.fit(y = data_train['sunspots'])


# Faz predição no conjunto de teste
print('Fazendo predição...')
predictions = forecaster.predict(steps = steps)

# Métricas de avaliação do modelo
y_true = data_test['sunspots']
y_pred = predictions

# RMSE
rmse = root_mean_squared_error(y_true, y_pred)
print('\nMétricas de avaliação do modelo')
print(f'RMSE: {rmse}')

# MAE
mae = mean_absolute_error(y_true, y_pred)
print(f'MAE: {mae}')

# Plota a predição, comparando com valores reais
fig, ax = plt.subplots()
data_train['sunspots'].plot(ax=ax, label='train')
data_test['sunspots'].plot(ax=ax, label='test')
predictions.plot(ax=ax, label='predictions', color='green')
ax.legend()
plt.show()