# Carregando os pacotes necessários para a análise em Python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from scipy.stats import shapiro
from scipy.stats import norm



# Carregando o data-frame (df) que contém os dados (formato Stata)
df = pd.read_stata("https://github.com/Daniel-Uhr/data/raw/main/cattaneo2.dta")


# print(df.to_csv('my_file.csv'))

# Vamos verificar as medidas de tendencia central (Média, Mediana e Moda) da variável 'bweight' (peso dos bebês ao nascer)
# print(df['bweight'].mean())
# print(df['bweight'].median())
# print(df['bweight'].mode())

# Vejamos agora as Medidas de Dispersão: Amplitude, Variância, e Desvio Padrão da variável 'bweight'
# print(df['bweight'].max() - df['bweight'].min())
# print(df['bweight'].var())
# print(df['bweight'].std())

# # Vamos verificar a distribuição da variável 'bweight' (peso dos bebês ao nascer) por meio de um histograma
# sns.histplot(df['bweight'], kde=True)

# # Vamos verificar a distribuição da variável 'bweight' (peso dos bebês ao nascer) 
# # por meio de um gráfico de densidade
# sns.kdeplot(df['bweight'], fill=True, label='Distribuição de bweight')
# # Vamos adicionar a média, mediana e moda ao gráfico
# plt.axvline(df['bweight'].mean(), color='red', linestyle='dashed', linewidth=1)
# plt.axvline(df['bweight'].median(), color='blue', linestyle='dashed', linewidth=1)
# plt.axvline(df['bweight'].mode()[0], color='green', linestyle='dashed', linewidth=1)

# # Agora vamos avaliar as medidas de forma: Assimetria e Curtose da variável 'bweight'
# print(df['bweight'].skew())
# print(df['bweight'].kurt())

# # Exibindo o gráfico
# plt.show()

# Plotando a densidade da variável 'bweight'
# sns.kdeplot(df['bweight'], fill=True, label='Distribuição de bweight')

# # Calculando a média e o desvio padrão da variável 'bweight'
# mean = np.mean(df['bweight'])
# std = np.std(df['bweight'])

# # Gerando uma sequência de valores para plotar a distribuição normal
# x_values = np.linspace(df['bweight'].min(), df['bweight'].max(), 1000)
# normal_dist = norm.pdf(x_values, mean, std)

# # Plotando a distribuição normal com uma linha pontilhada
# plt.plot(x_values, normal_dist, 'r--', label='Distribuição Normal')

# # # Exibindo o gráfico
# # plt.show()

# # # Criar Y e X, Y será igual a variável 'bweight' e X será 1 quando 'mbsmoke' for "smoker" e 0 caso contrário
df['X'] = (df['mbsmoke'] == 'smoker').astype(int)
df['Y'] = df['bweight']

# # # covariância entre Y e X (matriz de variância-covariância)
# # print(df[['Y', 'X']].cov())

# # # correlação entre Y e X
# # print(df[['Y', 'X']].corr())

# # Agora vamos calcular a média do peso dos bebês para fumantes e não fumantes
# # para não fumantes
# mean_n_fumantes = np.mean(df[df['X'] == 0]['Y'])
# # para fumantes
# mean_fumantes = np.mean(df[df['X'] == 1]['Y'])

# # print(mean_n_fumantes)
# # print(mean_fumantes)

# # Vejamos a distribuição do peso dos bebês para fumantes e não fumantes, marcando a média de cada grupo
# sns.kdeplot(df[df['X'] == 0]['Y'], fill=True, label='Não fumantes')
# sns.kdeplot(df[df['X'] == 1]['Y'], fill=True, label='Fumantes')

# # Adicionando linhas verticais para as médias
# plt.axvline(mean_n_fumantes, color='blue', linestyle='--', label='Média Não fumantes')
# plt.axvline(mean_fumantes, color='orange', linestyle='--', label='Média Fumantes')

# plt.show()

# Plotando a densidade da variável 'bweight'
sns.kdeplot(df['bweight'], fill=True, label='Distribuição de bweight')

# Calculando a média e o desvio padrão da variável 'bweight'
mean = np.mean(df['bweight'])
std = np.std(df['bweight'])

# Gerando uma sequência de valores para plotar a distribuição normal
x_values = np.linspace(df['bweight'].min(), df['bweight'].max(), 1000)
normal_dist = norm.pdf(x_values, mean, std)

# Plotando a distribuição normal com uma linha pontilhada
plt.plot(x_values, normal_dist, 'r--', label='Distribuição Normal')

# Exibindo o gráfico
# plt.show()


stat, p_value = shapiro(df['Y'])

print(f'Estatística W: {stat}')
print(f'Valor p: {p_value}')
# Verificando o resultado
if p_value > 0.5:
    print("Os dados parecem seguir uma distribuição normal (falha em rejeitar H0).")
else:
    print("Os dados não seguem uma distribuição normal (rejeita H0).")




result = stats.anderson(df['Y'], dist='norm')

print(f'Estatística: {result.statistic}')
print(f'Valores críticos: {result.critical_values}')
print(f'Significância: {result.significance_level}')


