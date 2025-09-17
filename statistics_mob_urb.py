import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pyreadr
import os

pontos_por_linha = {
    "343": {
        "A": [-43.31261, -23.00516],
        "B": [-43.19369, -22.90568]
    },
    "232": {
        "A": [-43.29042, -22.90076],
        "B": [-43.18941, -22.90825]
    },
    "390": {
        "A": [-43.39373, -22.95652],
        "B": [-43.19241, -22.90532]
    },
    "455": {
        "A": [-43.28039, -22.89977],
        "B": [-43.19089, -22.98671]
    },
    "600": {
        "A": [-43.40513, -22.91282],
        "B": [-43.22491, -22.92070]
    }
}

def getStatistics(filepath, df):

    df_ida = df[df['VECTOR_DIRECTION'] == 'Ida A → B']
    df_volta = df[df['VECTOR_DIRECTION'] == 'Volta B → A']

    lat_ida_mean = np.mean(df_ida['LATITUDE'])
    long_ida_mean = np.mean(df_ida['LONGITUDE'])
    lat_ida_median = np.median(df_ida['LATITUDE'])
    long_ida_median = np.median(df_ida['LONGITUDE'])

    lat_volta_mean = np.mean(df_volta['LATITUDE'])
    long_volta_mean = np.mean(df_volta['LONGITUDE'])
    lat_volta_median = np.median(df_volta['LATITUDE'])
    long_volta_median = np.median(df_volta['LONGITUDE'])

    lat_mean = np.mean(df['LATITUDE'])
    long_mean = np.mean(df['LONGITUDE'])
    lat_median = np.median(df['LATITUDE'])
    long_median = np.median(df['LONGITUDE'])

    velocity_mean = np.mean(df['VELOCITY'])

    lat_std = np.std(df['LATITUDE'])
    long_std = np.std(df['LONGITUDE'])
    velocity_std = np.std(df['VELOCITY'])

    print(lat_std)
    print(long_std)

    A = pontos_por_linha['455']['A']
    B = pontos_por_linha['455']['B']

            # Plot with statistics
    colors = ['green' if d == "Ida A → B" else 'red' if d == "Volta B → A" else 'blue' for d in df['VECTOR_DIRECTION']]
    plt.figure(figsize=(12, 6))
    plt.scatter(df['LONGITUDE'], df['LATITUDE'], c=colors, s=10)
    plt.plot(df['LONGITUDE'], df['LATITUDE'], linestyle='--', alpha=0.5)
    plt.scatter(*A, color='black', label='Ponto A (Chegada)', zorder=5)
    plt.scatter(*B, color='orange', label='Ponto B (Saída)', zorder=5)
    plt.title(f"Direção Instantânea (Produto Escalar) - {os.path.splitext(os.path.basename(filepath))[0]}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.axis('equal')
    plt.plot(long_ida_mean, lat_ida_mean, 'x', color='green', markersize=8)
    plt.plot(long_volta_mean, lat_volta_mean, 'x', color='red', markersize=8)
    plt.plot(long_mean, lat_mean, 'x', color='purple', markersize=8)
    plt.plot(long_ida_median, lat_ida_median, 'o', color='green', markersize=12)
    plt.plot(long_volta_median, lat_volta_median, 'o', color='red', markersize=12)
    plt.plot(long_median, lat_median, 'o', color='purple', markersize=12)
    plt.legend(handles=[
        mpatches.Patch(color='green', label='Ida A → B'),
        mpatches.Patch(color='red', label='Volta B → A'),
        mpatches.Patch(color='blue', label='Parado/Indefinido'),
        plt.Line2D([], [], marker='o', color='w', label='Ponto A (Chegada)', markerfacecolor='black', markersize=8),
        plt.Line2D([], [], marker='o', color='w', label='Ponto B (Saída)', markerfacecolor='orange', markersize=8)
    ])
    plt.show()