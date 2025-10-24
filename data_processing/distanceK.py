import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calcula a distância Haversine entre dois pontos na Terra.
    Retorna a distância em metros.
    """
    R = 6371000  # raio médio da Terra em metros
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def distancia_k(df, limite_ruido=7):
    """
    Calcula a distância acumulada (k) percorrida por cada ônibus da mesma linha.

    Distâncias menores que 'limite_ruido' metros são consideradas zero.
    """
    df = df[df['PARKING'].isna()]  # remove pontos de estacionamento (se houver coluna)

    line = df.iloc[0]['LINE']
    unique_ids = df['BUSID'].unique()

    # Seleciona no máximo 5 ônibus aleatórios
    sample_size = min(5, unique_ids.shape[0])
    rand_indices = np.random.choice(unique_ids.shape[0], size=sample_size, replace=False)
    sample_ids = unique_ids[rand_indices].astype(str)

    for busid in sample_ids:
        df_temp = df[df['BUSID'] == busid].copy()

        # Ordena por tempo
        df_temp = df_temp.sort_values(by='GPSTIMESTAMP')

        # Converte timestamp
        df_temp['GPSTIMESTAMP'] = pd.to_datetime(df_temp['GPSTIMESTAMP'])

        # Extrai lat/lon
        lat = df_temp['LATITUDE'].values
        lon = df_temp['LONGITUDE'].values

        # Distâncias entre pontos consecutivos
        dist_segmentos = haversine_np(lon[:-1], lat[:-1], lon[1:], lat[1:])

        # 🔹 Aplica limite: distâncias menores que 7m viram 0
        dist_segmentos = np.where(dist_segmentos < limite_ruido, 0, dist_segmentos)

        # Distância acumulada
        dist_acumulada = np.insert(np.cumsum(dist_segmentos), 0, 0)
        df_temp['DISTANCIA_ACUMULADA'] = dist_acumulada

        # Plotagem
        plt.figure(figsize=(12, 5))
        plt.plot(df_temp['GPSTIMESTAMP'], df_temp['DISTANCIA_ACUMULADA'], color='orange', label='Distância acumulada (m)')

        # Ajustes no eixo x
        start = df_temp['GPSTIMESTAMP'].min().replace(minute=0, second=0)
        end = df_temp['GPSTIMESTAMP'].max().replace(minute=0, second=0) + pd.Timedelta(hours=1)
        xticks = pd.date_range(start=start, end=end, freq='30min')
        plt.xticks(xticks, [t.strftime('%H:%M') for t in xticks], rotation=45)

        plt.title(f'Distância acumulada em função do tempo - Linha {line} (Ônibus {busid})')
        plt.xlabel('Hora')
        plt.ylabel('Distância acumulada (m)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()


# Exemplo de uso
if __name__ == "__main__":
    df = pd.read_csv('data/bus_csv/LINHA_432_COMPLETO.csv', sep=',')
    distancia_k(df, limite_ruido=7)
