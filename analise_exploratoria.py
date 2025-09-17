import numpy as np
import pandas as pd
import pyreadr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import glob
import os
from statistics_mob_urb import getStatistics

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

def haversine(lat1, lon1, lat2, lon2):
    """Calcula a distância Haversine entre dois pontos na Terra.

    Usa a fórmula de Haversine para determinar a distância geodésica entre dois pontos
    definidos por latitude e longitude, considerando a curvatura da Terra.

    Args:
        lat1 (float or np.ndarray): Latitude do primeiro ponto em graus.
        lon1 (float or np.ndarray): Longitude do primeiro ponto em graus.
        lat2 (float or np.ndarray): Latitude do segundo ponto em graus.
        lon2 (float or np.ndarray): Longitude do segundo ponto em graus.

    Returns:
        float or np.ndarray: Distância em quilômetros entre os pontos.

    Notes:
        O raio da Terra é fixado em 6.371 km.
    """
    R = 6371  # Raio da Terra em km
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def process_file(filepath):
    """Processa um arquivo de dados GPS para classificar direções e gerar visualizações.

    Determina os terminais A e B automaticamente com base nos extremos de latitude ou longitude,
    calcula direções vetoriais usando produto escalar e visualiza a tendência acumulada.

    Args:
        filepath (str): Caminho do arquivo RData com os dados GPS.

    Returns:
        None: Exibe três gráficos (direção instantânea, soma acumulada e mapa com tendência).

    Notes:
        Espera um DataFrame com colunas 'ID', 'LONGITUDE' e 'LATITUDE'.
        Os pontos A e B são definidos com base no maior deslocamento geográfico.
    """
    result = pyreadr.read_r(filepath)
    df: pd.DataFrame = result['filtered_data']
    df = df.sort_values(by='ID').reset_index(drop=True)

    # Calcular extremos de latitude e longitude
    min_lat_idx = df['LATITUDE'].idxmin()
    max_lat_idx = df['LATITUDE'].idxmax()
    min_lon_idx = df['LONGITUDE'].idxmin()
    max_lon_idx = df['LONGITUDE'].idxmax()

    min_lat_point = df.loc[min_lat_idx, ['LONGITUDE', 'LATITUDE']].values
    max_lat_point = df.loc[max_lat_idx, ['LONGITUDE', 'LATITUDE']].values
    min_lon_point = df.loc[min_lon_idx, ['LONGITUDE', 'LATITUDE']].values
    max_lon_point = df.loc[max_lon_idx, ['LONGITUDE', 'LATITUDE']].values

    # Calcular distâncias entre extremos
    dist_lat = haversine(min_lat_point[1], min_lat_point[0], max_lat_point[1], max_lat_point[0])
    dist_lon = haversine(min_lon_point[1], min_lon_point[0], max_lon_point[1], max_lon_point[0])

    # Definir A e B com base no maior deslocamento geográfico
    if dist_lat > dist_lon:
        A = min_lat_point
        B = max_lat_point
    else:
        A = min_lon_point
        B = max_lon_point

    vector_AB = B - A

    vector_directions = []
    for i in range(len(df)):
        if i == 0 or i == len(df) - 1:
            vector_directions.append("NA")
            continue
        next_point = np.array([df.iloc[i + 1]['LONGITUDE'], df.iloc[i + 1]['LATITUDE']])
        current_point = np.array([df.iloc[i]['LONGITUDE'], df.iloc[i]['LATITUDE']])
        movement_vector = next_point - current_point
        dot_product = np.dot(movement_vector, vector_AB)
        if dot_product > 0:
            direction = "Ida A → B"
        elif dot_product < 0:
            direction = "Volta B → A"
        else:
            direction = "Parado/Perpendicular"
        vector_directions.append(direction)
    df['VECTOR_DIRECTION'] = vector_directions

    direction_numerical = df['VECTOR_DIRECTION'].map({
        'Ida A → B': 1,
        'Volta B → A': -1
    }).fillna(0)
    df['DIRECTION_CUMSUM'] = direction_numerical.cumsum()

    def interpret_trend(val):
        if val > 0:
            return 'Tendência: Ida'
        elif val < 0:
            return 'Tendência: Volta'
        else:
            return 'Tendência: Neutra'
    df['TREND_DIRECTION'] = df['DIRECTION_CUMSUM'].apply(interpret_trend)

    getStatistics(filepath,df)

    # # Plot 1
    # colors = ['green' if d == "Ida A → B" else 'red' if d == "Volta B → A" else 'blue' for d in df['VECTOR_DIRECTION']]
    # plt.figure(figsize=(12, 6))
    # plt.scatter(df['LONGITUDE'], df['LATITUDE'], c=colors, s=10)
    # plt.plot(df['LONGITUDE'], df['LATITUDE'], linestyle='--', alpha=0.5)
    # plt.scatter(*A, color='black', label='Ponto A (Chegada)', zorder=5)
    # plt.scatter(*B, color='orange', label='Ponto B (Saída)', zorder=5)
    # plt.title(f"Direção Instantânea (Produto Escalar) - {os.path.splitext(os.path.basename(filepath))[0]}")
    # plt.xlabel("Longitude")
    # plt.ylabel("Latitude")
    # plt.grid(True)
    # plt.axis('equal')
    # plt.legend(handles=[
    #     mpatches.Patch(color='green', label='Ida A → B'),
    #     mpatches.Patch(color='red', label='Volta B → A'),
    #     mpatches.Patch(color='blue', label='Parado/Indefinido'),
    #     plt.Line2D([], [], marker='o', color='w', label='Ponto A (Chegada)', markerfacecolor='black', markersize=8),
    #     plt.Line2D([], [], marker='o', color='w', label='Ponto B (Saída)', markerfacecolor='orange', markersize=8)
    # ])
    # plt.show()

    # # Plot 2
    # plt.figure(figsize=(12, 4))
    # plt.plot(df['DIRECTION_CUMSUM'], label='Soma acumulada da direção', color='purple')
    # plt.axhline(0, color='gray', linestyle='--')
    # plt.title(f"Tendência Acumulada da Direção (Cumsum) - {os.path.splitext(os.path.basename(filepath))[0]}")
    # plt.xlabel("Índice do ponto")
    # plt.ylabel("Acúmulo da direção")
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    # # Plot 3
    # trend_colors = df['TREND_DIRECTION'].map({
    #     'Tendência: Ida': 'green',
    #     'Tendência: Volta': 'red',
    #     'Tendência: Neutra': 'gray'
    # })
    # plt.figure(figsize=(12, 6))
    # plt.scatter(df['LONGITUDE'], df['LATITUDE'], c=trend_colors, s=10)
    # plt.plot(df['LONGITUDE'], df['LATITUDE'], linestyle='--', alpha=0.5)
    # plt.scatter(*A, color='black', label='Ponto A (Chegada)', zorder=5)
    # plt.scatter(*B, color='orange', label='Ponto B (Saída)', zorder=5)
    # plt.title(f"Mapa com Tendência Direcional Acumulada - {os.path.splitext(os.path.basename(filepath))[0]}")
    # plt.xlabel("Longitude")
    # plt.ylabel("Latitude")
    # plt.grid(True)
    # plt.axis('equal')
    # plt.legend(handles=[
    #     mpatches.Patch(color='green', label='Tendência: Ida'),
    #     mpatches.Patch(color='red', label='Tendência: Volta'),
    #     mpatches.Patch(color='gray', label='Tendência: Neutra'),
    #     plt.Line2D([], [], marker='o', color='w', label='Ponto A (Chegada)', markerfacecolor='black', markersize=8),
    #     plt.Line2D([], [], marker='o', color='w', label='Ponto B (Saída)', markerfacecolor='orange', markersize=8)
    # ])
    # plt.show()

if __name__ == "__main__":

    # Obtém o caminho absoluto do diretório onde o script está
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(dir_path, "linha_455_ida_volta.RData")
    
    # Processa o arquivo
    process_file(file_path)