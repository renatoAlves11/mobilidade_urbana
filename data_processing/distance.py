import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def distance(lat_origin, lon_origin, df):
    print(f'Lat origin: {lat_origin}')
    print(f'Long origin: {lon_origin}')
    
    line = df.iloc[0]['LINE']
    unique_ids = df['BUSID'].unique()
    sample_size = min(5, unique_ids.shape[0])
    randNum = np.random.choice(unique_ids.shape[0], size=sample_size, replace=False)
    sample_ids = unique_ids[randNum].astype(str)
    df = df[df['PARKING'].isna()]

    for j in range(sample_size):
        busid = sample_ids[j]
        df_temp = df[df['BUSID'] == busid]

        if len(df_temp) < 1000:
            continue

        distances = haversine_np(
            lon_origin,
            lat_origin,
            df_temp['LONGITUDE'].values,
            df_temp['LATITUDE'].values
        )

        # Converte GPS_TIMESTAMP para datetime
        timestamps = pd.to_datetime(df_temp['GPSTIMESTAMP'])

        plt.figure(figsize=(12, 5))
        plt.plot(timestamps, distances, label='Distância (m)', color='royalblue')

        # Cria ticks a cada 30 minutos
        start = timestamps.min().replace(minute=0, second=0)
        end = timestamps.max().replace(minute=0, second=0) + pd.Timedelta(hours=1)
        xticks = pd.date_range(start=start, end=end, freq='30min')
        plt.xticks(xticks, [t.strftime('%H:%M') for t in xticks], rotation=45)

        plt.title(f'Módulo das distâncias em função do tempo da linha {line} (ônibus {busid})')
        plt.xlabel('Hora')
        plt.ylabel('Distância (m)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()
