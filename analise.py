import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import folium
from shapely.geometry import Point
from shapely.affinity import scale

# Pontos fixos
pontos_por_linha = {
    "343": {"A": [-43.31261, -23.00516], "B": [-43.19369, -22.90568]},
    "232": {"A": [-43.29042, -22.90076], "B": [-43.18941, -22.90825]},
    "390": {"A": [-43.39373, -22.95652], "B": [-43.19241, -22.90532]},
    "455": {"A": [-43.28039, -22.89977], "B": [-43.19089, -22.98671]},
    "600": {"A": [-43.40513, -22.91282], "B": [-43.22491, -22.92070]}
}

# Lê CSV
df = pd.read_csv("arquivo_unificado.csv")

# Gera dfs para coleta de estatísticas

df_ida = df[df['SEGMENT_CLASSIFICATION'] == 'Ida']
df_volta = df[df['SEGMENT_CLASSIFICATION'] == 'Volta']

#Coletar estátisticas para análise
theta = np.linspace(0, 2*np.pi, 100) 

lat_ida_mean = np.mean(df_ida['LATITUDE'])
long_ida_mean = np.mean(df_ida['LONGITUDE'])

lat_volta_mean = np.mean(df_volta['LATITUDE'])
long_volta_mean = np.mean(df_volta['LONGITUDE'])

lat_mean = np.mean(df['LATITUDE'])
long_mean = np.mean(df['LONGITUDE'])

lat_std = np.std(df['LATITUDE'])
long_std = np.std(df['LONGITUDE'])
lat_ida_std = np.std(df_ida['LATITUDE'])
long_ida_std = np.std(df_ida['LONGITUDE'])
lat_volta_std = np.std(df_volta['LATITUDE'])
long_volta_std = np.std(df_volta['LONGITUDE'])

# ====== 1. PLOT GERAL ======
fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(df["LONGITUDE"], df["LATITUDE"], s=10, c="purple", alpha=0.6, label="Geral")

for i in range(len(df)-1):
    # pega classificação (ou outro critério) de cada ponto
    class1 = df.iloc[i]["LINE"]
    class2 = df.iloc[i+1]["LINE"]
    
    # só conecta se forem iguais
    if class1 == class2:
        lat1, lon1 = df.iloc[i][["LATITUDE", "LONGITUDE"]]
        lat2, lon2 = df.iloc[i+1][["LATITUDE", "LONGITUDE"]]
        ax.plot([lon1, lon2], [lat1, lat2], color="purple", linewidth=1, alpha=0.6)

linha = "232"

# Pontos fixos A e B
ax.scatter([pontos_por_linha[linha]["A"][0]], [pontos_por_linha[linha]["A"][1]], color='yellow', s=100, label="Ponto A", zorder=5)
ax.scatter([pontos_por_linha[linha]["B"][0]], [pontos_por_linha[linha]["B"][1]], color='black', s=100, label="Ponto B", zorder=5)

# Elipse 
x_ellipse = long_mean + long_std * np.cos(theta)
y_ellipse = lat_mean + lat_std * np.sin(theta)
ax.scatter(long_mean, lat_mean, color='purple', s=100, label="Geral", zorder=5, marker='x')
ax.plot(x_ellipse, y_ellipse, color='purple', linestyle='--', linewidth=1.5, label=f"Elipse Geral")

ax.set_xlabel("LONGITUDE")
ax.set_ylabel("LATITUDE")
ax.set_title("Trajetória Geral")
ax.legend(loc='lower left')

plt.tight_layout()
plt.show()


# # ====== 2. PLOT IDA vs VOLTA DINÂMICO ======
# fig, ax = plt.subplots(figsize=(10, 6))

# # Mapa de cores
# colors_map = {"Ida": "green", "Volta": "red", "Indefinido": "gray"}

# # Pontos coloridos dinamicamente
# for i in range(len(df)):
#     classificacao = df.iloc[i]["SEGMENT_CLASSIFICATION"]
#     cor = colors_map.get(classificacao, "blue")

#     ax.scatter(df.iloc[i]["LONGITUDE"], df.iloc[i]["LATITUDE"],
#                s=10, c=cor, alpha=0.6)

# # Linhas entre pontos consecutivos (também coloridas dinamicamente)
# for i in range(len(df)-1):
#     classificacao = df.iloc[i]["SEGMENT_CLASSIFICATION"]
#     cor = colors_map.get(classificacao, "blue")

#     lat1, lon1 = df.iloc[i][["LATITUDE", "LONGITUDE"]]
#     lat2, lon2 = df.iloc[i+1][["LATITUDE", "LONGITUDE"]]
#     ax.plot([lon1, lon2], [lat1, lat2], color=cor, linewidth=1, alpha=0.6)

# # Pontos fixos A e B
# ax.scatter([pontos_por_linha[linha]["A"][0]], [pontos_por_linha[linha]["A"][1]], color='yellow', s=100, label="Ponto A", zorder=5)
# ax.scatter([pontos_por_linha[linha]["B"][0]], [pontos_por_linha[linha]["B"][1]], color='black', s=100, label="Ponto B", zorder=5)
# ax.scatter(long_ida_mean, lat_ida_mean, color='green', s=100, label="Ida", zorder=5, marker='x')
# ax.scatter(long_volta_mean, lat_volta_mean, color='red', s=100, label="Volta", zorder=5, marker='x')

# # # Elipse 
# # x_ellipse = long_ida_mean + long_ida_std * np.cos(theta)
# # y_ellipse = lat_ida_mean + lat_ida_std * np.sin(theta)
# # # ax.plot(x_ellipse, y_ellipse, color='green', linestyle='--', linewidth=1.5, label=f"Elipse Ida")

# # x_ellipse = long_volta_mean + long_volta_std * np.cos(theta) 
# # y_ellipse = lat_volta_mean + lat_volta_std * np.sin(theta) 
# # # ax.plot(x_ellipse, y_ellipse, color='red', linestyle='--', linewidth=1.5, label=f"Elipse Volta")

# # ax.set_xlabel("LONGITUDE")
# # ax.set_ylabel("LATITUDE")
# # ax.set_title("Trajetórias Ida e Volta (dinâmico)")
# # ax.legend(loc='lower left')

# # plt.tight_layout()
# # plt.show()

# === Folium Map ===
m = folium.Map(location=[lat_mean, long_mean], zoom_start=12)

# # ---- Trajetória Geral ----
# folium.PolyLine(
#     df[['LATITUDE','LONGITUDE']].values.tolist(),
#     color="purple", weight=2, opacity=0.6,
#     tooltip="Trajetória Geral"
# ).add_to(m)

# Mapeamento de cores
colors_map = {"Ida": "green", "Volta": "red", "Indefinido": "gray"}

# Loop em pares consecutivos
for i in range(len(df) - 1):

    class1 = df.iloc[i]["LINE"]
    class2 = df.iloc[i+1]["LINE"]

    if class1 == class2:

        classificacao = df.iloc[i]["SEGMENT_CLASSIFICATION"]
        cor = colors_map.get(classificacao, "blue")  # default azul se não tiver mapeado
        
        lat1, lon1 = df.iloc[i][["LATITUDE", "LONGITUDE"]]
        lat2, lon2 = df.iloc[i+1][["LATITUDE", "LONGITUDE"]]
        
        folium.PolyLine(
            [(lat1, lon1), (lat2, lon2)],
            color=cor, weight=2, opacity=0.6
        ).add_to(m)

# ---- Pontos Fixos A e B de todas as linhas ----
for linha_key, pontos in pontos_por_linha.items():
    # Ponto A
    folium.Marker(
        location=[pontos["A"][1], pontos["A"][0]],  # folium usa [lat, lon]
        popup=f"Ponto A - Linha {linha_key}",
        icon=folium.Icon(color="yellow", icon="flag")
    ).add_to(m)
    
    # Ponto B
    folium.Marker(
        location=[pontos["B"][1], pontos["B"][0]],
        popup=f"Ponto B - Linha {linha_key}",
        icon=folium.Icon(color="black", icon="flag")
    ).add_to(m)


# ---- Médias ----
folium.Marker([lat_mean, long_mean], popup="Média Geral",
              icon=folium.Icon(color="purple", icon="x")).add_to(m)

folium.Marker([lat_ida_mean, long_ida_mean], popup="Média Ida",
              icon=folium.Icon(color="green", icon="x")).add_to(m)

folium.Marker([lat_volta_mean, long_volta_mean], popup="Média Volta",
              icon=folium.Icon(color="red", icon="x")).add_to(m)

# ---- Função para desenhar elipse ----
def add_ellipse(map_obj, lat, lon, std_lat, std_lon, color, name, n_points=100):
    circle = Point(lon, lat).buffer(1, resolution=n_points)
    ellipse = scale(circle, std_lon, std_lat)  
    ellipse_coords = [(y, x) for x, y in ellipse.exterior.coords]
    folium.PolyLine(ellipse_coords, color=color, weight=2, dash_array="5,5",
                    tooltip=f"Elipse {name}").add_to(map_obj)

# Adiciona elipses
add_ellipse(m, lat_mean, long_mean, lat_std, long_std, "purple", "Geral")
add_ellipse(m, lat_ida_mean, long_ida_mean, lat_ida_std, long_ida_std, "green", "Ida")
add_ellipse(m, lat_volta_mean, long_volta_mean, lat_volta_std, long_volta_std, "red", "Volta")

# ---- Salva HTML ----
m.save("mapa_interativo.html")