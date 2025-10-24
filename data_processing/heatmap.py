import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
from distance import distance
from compare_distances import comparar_distancias

# Caminho do diretório com os CSVs
diretorio = "data/bus_csv/sul"

# Lista todos os arquivos CSV no diretório
arquivos_csv = glob.glob(os.path.join(diretorio, "*.csv"))

# Cria um array/lista de DataFrames
dataframes = [pd.read_csv(arquivo) for arquivo in arquivos_csv]

for df in dataframes:

    # Filtra apenas ônibus que não estão em garagem
    df = df[df['PARKING'].isna()]

    line = df.iloc[0]['LINE']

    round_val = 3

    df['LATITUDE'] = df['LATITUDE'].round(round_val)
    df['LONGITUDE'] = df['LONGITUDE'].round(round_val)

    # Cria a tabela de frequência (latitudes nas linhas, longitudes nas colunas)
    matriz = pd.crosstab(df['LATITUDE'], df['LONGITUDE'])

    # Encontra o valor máximo da matriz
    max_freq = matriz.values.max()

    # Define o intervalo de tolerância (por exemplo, até 3 pontos abaixo do máximo)
    tolerancia = 3
    limite_inferior = max_freq - tolerancia

    # Localiza todas as posições cuja frequência está dentro desse intervalo
    posicoes_intervalo = np.argwhere((matriz.values >= limite_inferior) & (matriz.values <= max_freq))

    # Converte os índices numéricos para os rótulos reais (longitudes e latitudes)
    longitudes = matriz.index[posicoes_intervalo[:, 0]]
    latitudes = matriz.columns[posicoes_intervalo[:, 1]]
    valores = matriz.values[posicoes_intervalo[:, 0], posicoes_intervalo[:, 1]]

    print(matriz)
    print(f'Frequência máxima encontrada: {max_freq}')
    print(f'Coordenadas com frequência entre {limite_inferior} e {max_freq}:')

    for lon, lat, val in zip(latitudes, longitudes, valores):
        print(f'  → Longitude: {lon}, Latitude: {lat}, Frequência: {val}')

    # Plot do heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matriz[::-1],
        cmap='Blues',
        cbar_kws={'label': 'Frequência'},
        square=True
    )

    plt.title(f'Mapa de Calor - Frequência de posições (Lat x Lon) da linha {line}\n(Valores entre {limite_inferior} e {max_freq})')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

    distance(matriz.index[0], matriz.columns[0], df)