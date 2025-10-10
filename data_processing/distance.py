import numpy as np
import matplotlib.pyplot as plt

def haversine_np(lon1, lat1, lon2, lat2):
    """Calcula a distância Haversine entre dois pontos na Terra.

    Usa a fórmula de Haversine para determinar a distância geodésica entre dois pontos
    definidos por longitude e latitude, considerando a curvatura da Terra.

    Args:
        lon1 (float or np.ndarray): Longitude do primeiro ponto em graus.
        lat1 (float or np.ndarray): Latitude do primeiro ponto em graus.
        lon2 (float or np.ndarray): Longitude do segundo ponto em graus.
        lat2 (float or np.ndarray): Latitude do segundo ponto em graus.

    Returns:
        float or np.ndarray: Distância em metros entre os pontos.

    Notes:
        O raio da Terra é fixado em 6.371.000 metros.
    """
    R = 6371000
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def distance(lat_origin, lon_origin, df):

    line = df.iloc[0]['LINE']
    unique_ids = df['BUSID'].unique()
    sample_size = min(5, unique_ids.shape[0])
    randNum = np.random.choice(unique_ids.shape[0], size=sample_size, replace=False)
    sample_ids = unique_ids[randNum].astype(str)

    for j in range(sample_size):
        busid = sample_ids[j]

        df_temp = df[(df['BUSID'] == busid)]

        if len(df_temp) < 1000:
            continue

        distances = []

        for i in range(len(df_temp)):
            lat = df_temp.iloc[i]['LATITUDE']
            lon = df_temp.iloc[i]['LONGITUDE']
            dist = haversine_np(lon_origin, lat_origin, lon, lat)
            distances.append(np.abs(dist))

        t = np.arange(len(df_temp))  # tempo de 0 até len(df)-1
        plt.figure(figsize=(10, 5))
        plt.plot(t, distances, label='Distância (m)', color='royalblue')
        plt.title(f'Módulo das distâncias em função do tempo t da linha {line} e ônibus {busid}')
        plt.xlabel('Tempo (t)')
        plt.ylabel('Distância (m)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.show()