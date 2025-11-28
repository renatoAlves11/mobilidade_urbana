import numpy as np
import pandas as pd
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

def detectar_paradas(df_temp, dist_ponto, dist_acumulada, t=5, tol=2.0):
    """
    Detecta períodos de parada com base em variações pequenas em dist_ponto e dist_acumulada.
    
    Parâmetros:
      df_temp: DataFrame com LATITUDE e LONGITUDE
      dist_ponto: array com distância ao ponto fixo
      dist_acumulada: array com distância acumulada
      t: número mínimo de amostras consecutivas
      tol: variação máxima (em metros) considerada 'constante'
      
    Retorna:
      lista de tuplas (lat_médio, lon_médio)
    """
    lat = df_temp['LATITUDE'].values
    lon = df_temp['LONGITUDE'].values

    # suaviza pequenas flutuações (rolling window)
    dist_ponto_smooth = pd.Series(dist_ponto).rolling(window=3, center=True, min_periods=1).mean()
    dist_k_smooth = pd.Series(dist_acumulada).rolling(window=3, center=True, min_periods=1).mean()

    delta_ponto = np.abs(np.diff(dist_ponto_smooth))
    delta_k = np.abs(np.diff(dist_k_smooth))

    # condição de constância
    const_mask = (delta_ponto < tol) & (delta_k < tol)

    pontos_constantes = []
    count = 0

    # varre sequência para achar blocos de t consecutivos
    for i in range(len(const_mask)):
        if const_mask[i]:
            count += 1
        else:
            if count >= t:
                idx_ini = i - count
                idx_fim = i
                lat_med = np.mean(lat[idx_ini:idx_fim])
                lon_med = np.mean(lon[idx_ini:idx_fim])
                pontos_constantes.append((lat_med, lon_med))
            count = 0

    # caso termine em sequência constante
    if count >= t:
        lat_med = np.mean(lat[-count:])
        lon_med = np.mean(lon[-count:])
        pontos_constantes.append((lat_med, lon_med))

    return pontos_constantes


def comparar_distancias(lat_origin, lon_origin, df, limite_ruido=7, t=3, tol=2.0):
    df = df[df['PARKING'].isna()]
    line = df.iloc[0]['LINE']
    unique_ids = df['BUSID'].unique()
    sample_size = min(5, unique_ids.shape[0])
    sample_ids = np.random.choice(unique_ids, size=sample_size, replace=False)

    for busid in sample_ids:
        df_temp = df[df['BUSID'] == busid].copy()
        if len(df_temp) < 1000:
            continue

        df_temp = df_temp.sort_values(by='GPSTIMESTAMP')
        df_temp['GPSTIMESTAMP'] = pd.to_datetime(df_temp['GPSTIMESTAMP'])
        timestamps = df_temp['GPSTIMESTAMP']

        lat = df_temp['LATITUDE'].values
        lon = df_temp['LONGITUDE'].values

        # distância até o ponto fixo
        dist_ponto = haversine_np(lon_origin, lat_origin, lon, lat)

        # distância acumulada
        dist_segmentos = haversine_np(lon[:-1], lat[:-1], lon[1:], lat[1:])
        dist_segmentos = np.where(dist_segmentos < limite_ruido, 0, dist_segmentos)
        dist_acumulada = np.insert(np.cumsum(dist_segmentos), 0, 0)

        # detectar pontos de parada
        pontos_parada = detectar_paradas(df_temp, dist_ponto, dist_acumulada, t=t, tol=tol)

        # plot
        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        axs[0].plot(timestamps, dist_ponto, label='Distância ao ponto inicial (m)', color='royalblue')
        axs[0].set_title(f'Distância até o ponto fixo - Linha {line} (Ônibus {busid})')
        axs[0].set_ylabel('Distância (m)')
        axs[0].grid(True, linestyle='--', alpha=0.7)
        axs[0].legend()

        axs[1].plot(timestamps, dist_acumulada, label='Distância acumulada (m)', color='orange')
        axs[1].set_title('Distância acumulada ao longo do tempo')
        axs[1].set_xlabel('Hora')
        axs[1].set_ylabel('Distância acumulada (m)')
        axs[1].grid(True, linestyle='--', alpha=0.7)
        axs[1].legend()

        # marca paradas no segundo gráfico
        for (lat_p, lon_p) in pontos_parada:
            i = np.argmin(np.abs(lat - lat_p) + np.abs(lon - lon_p))
            axs[1].axvline(timestamps.iloc[i], color='red', linestyle='--', alpha=0.5)

        # Cria ticks a cada 30 minutos
        start = timestamps.min().replace(minute=0, second=0)
        end = timestamps.max().replace(minute=0, second=0) + pd.Timedelta(hours=1)
        xticks = pd.date_range(start=start, end=end, freq='30min')
        plt.xticks(xticks, [t.strftime('%H:%M') for t in xticks], rotation=45)
        plt.tight_layout()
        plt.show()

        print(f"\nLinha: {line}")
        print(f"\nPontos de parada detectados para ônibus {busid}:")
        for lat_p, lon_p in pontos_parada:
            print(f"  -> lat={lat_p:.6f}, lon={lon_p:.6f}")


# Exemplo de uso:
if __name__ == "__main__":
    df = pd.read_csv('../data/bus_csv/sul/LINHA_432_COMPLETO.csv', sep=',')
    lat_origin, lon_origin = -22.905, -43.176
    comparar_distancias(lat_origin, lon_origin, df, limite_ruido=7, t=5, tol=10.0)
