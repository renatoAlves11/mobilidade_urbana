import numpy as np
import pandas as pd
import pyreadr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import glob
import os
import seaborn as sns
from pandas.plotting import parallel_coordinates

def process_file(file_path):
    # Ler o CSV
    df = pd.read_csv(file_path)

    # Colunas de interesse
    cols = ["LATITUDE", "LONGITUDE", "VELOCITY", "ELEVATION", "RAINFALLZONE", "RAINFALLVOLUME", "SPEED", "CO_2", "CO"]

    # Média e desvio padrão geral
    estatisticas_geral = df[cols].agg(['mean', 'std'])
    print("Estatísticas gerais (média e desvio padrão):")
    print(estatisticas_geral)
    print("\n")

    # Média e desvio padrão agrupado por LINE
    estatisticas_por_line = df.groupby("LINE")[cols].agg(['mean', 'std']).reset_index()
    print("Estatísticas por LINE (média e desvio padrão):")
    print(estatisticas_por_line)

    return estatisticas_geral, estatisticas_por_line

def plot_distributions_and_boxplots(file_path):
    # Ler CSV
    df = pd.read_csv(file_path)

    # Colunas de interesse
    cols = ["LATITUDE", "LONGITUDE", "VELOCITY", "ELEVATION", "RAINFALLZONE", "RAINFALLVOLUME", "SPEED", "CO_2", "CO"]

    for col in cols:
        plt.figure(figsize=(12,5))

        # Gráfico de densidade (Distribution / KDE)
        plt.subplot(1,2,1)
        sns.kdeplot(data=df, x=col, hue="LINE", fill=True, alpha=0.4)
        plt.title(f'Distribuição de {col} por LINE')

        # Boxplot
        plt.subplot(1,2,2)
        sns.boxplot(data=df, x="LINE", y=col)
        plt.title(f'Boxplot de {col} por LINE')

        plt.tight_layout()
        plt.show()

def plot_correlation(file_path):
    # Ler CSV
    df = pd.read_csv(file_path)

    # Selecionar colunas numéricas de interesse
    cols = ["VELOCITY", "ELEVATION", "RAINFALLZONE", "RAINFALLVOLUME", "SPEED", "CO_2", "CO", "LATITUDE", "LONGITUDE"]
    df_num = df[cols]

    # Calcular a correlação
    corr = df_num.corr()  # Pearson por padrão

    # Plotar o heatmap
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Mapa de Correlação entre os atributos")
    plt.show()

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def plot_parallel_coordinates(df):
    cols = ["VELOCITY", "ELEVATION", "RAINFALLZONE", "RAINFALLVOLUME", "SPEED", "CO_2", "CO"]
    df_norm = (df[cols] - df[cols].min()) / (df[cols].max() - df[cols].min())
    
    # Criar um mapa de cores para cada LINE
    unique_lines = df['LINE'].unique()
    colors = sns.color_palette("tab10", n_colors=len(unique_lines))
    line_color_map = {line: colors[i % len(colors)] for i, line in enumerate(unique_lines)}
    
    plt.figure(figsize=(12,6))
    for i in range(len(df_norm)):
        plt.plot(df_norm.columns, df_norm.iloc[i], color=line_color_map[df['LINE'].iloc[i]], alpha=0.5)
    
    # Criar legenda
    handles = [plt.Line2D([0], [0], color=line_color_map[line], lw=2) for line in unique_lines]
    plt.legend(handles, unique_lines, title="LINE", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.title("Coordenadas Paralelas por LINE")
    plt.xlabel("Atributos")
    plt.ylabel("Valores normalizados")
    plt.show()

# Quiver plot para todos os dados
def plot_quiver(df):
    plt.figure(figsize=(8,6))
    u = np.cos(np.radians(df['VELOCITY']))  # direção x simulada
    v = np.sin(np.radians(df['VELOCITY']))  # direção y simulada
    plt.quiver(df['LONGITUDE'], df['LATITUDE'], u, v, df['VELOCITY'], scale=50, cmap='viridis')
    plt.colorbar(label='VELOCITY')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Orientação por Pixel (Quiver Plot) - todos os dados")
    plt.show()

# Função principal
def plot_all(df):
    plot_parallel_coordinates(df)
    plot_quiver(df)

# estatisticas_geral, estatisticas_por_line = process_file("arquivo_unificado.csv")
# plot_distributions_and_boxplots("arquivo_unificado.csv")
# plot_correlation("arquivo_unificado.csv")
df = pd.read_csv("arquivo_unificado.csv")
plot_all(df)
