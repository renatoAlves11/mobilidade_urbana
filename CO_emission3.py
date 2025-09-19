from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import folium
import pyreadr
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as colors

# === Carregar todos os arquivos RData ===
arquivos = glob.glob("linha_*_ida_volta.RData")

dfs = []
for arquivo in arquivos:
    result = pyreadr.read_r(arquivo)
    df = list(result.values())[0]

    # Seleciona colunas importantes
    df = df[["GPSTIMESTAMP", "LINE", "CO_2", "VELOCITY", "LATITUDE", "LONGITUDE",
             "NEIGHBORHOOD", "ADMINISTRATIVEREGION", "RAINFALLVOLUME"]].dropna()

    # Converte timestamp
    df["GPSTIMESTAMP"] = pd.to_datetime(df["GPSTIMESTAMP"], errors="coerce")
    df["HORA"] = df["GPSTIMESTAMP"].dt.hour

    # Filtra valores inválidos
    df = df[df["CO_2"] > 0]

    # Adiciona info da linha a partir do próprio arquivo
    df["LINE_FILE"] = arquivo.split("_")[1]

    dfs.append(df)

# Junta tudo em um único DataFrame
dados = pd.concat(dfs, ignore_index=True)
print("Dados carregados:", dados["LINE_FILE"].unique())

import folium

# Criar mapa centrado no RJ
mapa = folium.Map(location=[-22.9068, -43.1729], zoom_start=11)

# Normalizar CO2 entre 0 e 1
dados['CO2_norm'] = (dados['CO_2'] - dados['CO_2'].min()) / (dados['CO_2'].max() - dados['CO_2'].min())

# Função para mapear normalizado para cor (amarelo -> vermelho)
def cor_custom(valor_norm):
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    # Criar colormap customizado
    cmap = cm.get_cmap('YlOrRd')  # YlOrRd vai de amarelo a vermelho
    return mcolors.to_hex(cmap(valor_norm))

# Adicionar círculos
for _, row in dados.iterrows():
    folium.CircleMarker(
        location=[row['LATITUDE'], row['LONGITUDE']],
        radius=5 + row['CO2_norm']*15,  # tamanho proporcional
        color=cor_custom(row['CO2_norm']),
        fill=True,
        fill_color=cor_custom(row['CO2_norm']),
        fill_opacity=0.8,
        popup=f"CO₂: {row['CO_2']}"
    ).add_to(mapa)

mapa.save("mapa_CO2_emissao.html")