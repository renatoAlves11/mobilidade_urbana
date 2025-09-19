import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Pontos fixos
pontos_por_linha = {
    "343": {"A": [-43.31261, -23.00516], "B": [-43.19369, -22.90568]},
    "232": {"A": [-43.29042, -22.90076], "B": [-43.18941, -22.90825]},
    "390": {"A": [-43.39373, -22.95652], "B": [-43.19241, -22.90532]},
    "455": {"A": [-43.28039, -22.89977], "B": [-43.19089, -22.98671]},
    "600": {"A": [-43.40513, -22.91282], "B": [-43.22491, -22.92070]}
}

# Lê CSV
df = pd.read_csv("resultado_segmentacao_LINHA_232_B25542_COMPLETO.csv")

# ====== 1. PLOT GERAL ======
fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(df["LONGITUDE"], df["LATITUDE"], s=10, c="purple", alpha=0.6, label="Geral")

for i in range(len(df)-1):
    lat1, lon1 = df.iloc[i][["LATITUDE", "LONGITUDE"]]
    lat2, lon2 = df.iloc[i+1][["LATITUDE", "LONGITUDE"]]
    ax.plot([lon1, lon2], [lat1, lat2], color="purple", linewidth=1, alpha=0.6)

linha = "232"

# Pontos fixos A e B
ax.scatter([pontos_por_linha[linha]["A"][0]], [pontos_por_linha[linha]["A"][1]], color='yellow', s=100, label="Ponto A", zorder=5)
ax.scatter([pontos_por_linha[linha]["B"][0]], [pontos_por_linha[linha]["B"][1]], color='black', s=100, label="Ponto B", zorder=5)

ax.set_xlabel("LONGITUDE")
ax.set_ylabel("LATITUDE")
ax.set_title("Trajetória Geral")
ax.legend(loc='lower left')

plt.tight_layout()
plt.show()


# ====== 2. PLOT IDA vs VOLTA DINÂMICO ======
fig, ax = plt.subplots(figsize=(10, 6))

# Mapa de cores
colors_map = {"Ida": "green", "Volta": "red", "Indefinido": "gray"}

# Pontos coloridos dinamicamente
for i in range(len(df)):
    classificacao = df.iloc[i]["SEGMENT_CLASSIFICATION"]
    cor = colors_map.get(classificacao, "blue")

    ax.scatter(df.iloc[i]["LONGITUDE"], df.iloc[i]["LATITUDE"],
               s=10, c=cor, alpha=0.6)

# Linhas entre pontos consecutivos (também coloridas dinamicamente)
for i in range(len(df)-1):
    classificacao = df.iloc[i]["SEGMENT_CLASSIFICATION"]
    cor = colors_map.get(classificacao, "blue")

    lat1, lon1 = df.iloc[i][["LATITUDE", "LONGITUDE"]]
    lat2, lon2 = df.iloc[i+1][["LATITUDE", "LONGITUDE"]]
    ax.plot([lon1, lon2], [lat1, lat2], color=cor, linewidth=1, alpha=0.6)

# Pontos fixos A e B
ax.scatter([pontos_por_linha[linha]["A"][0]], [pontos_por_linha[linha]["A"][1]], color='yellow', s=100, label="Ponto A", zorder=5)
ax.scatter([pontos_por_linha[linha]["B"][0]], [pontos_por_linha[linha]["B"][1]], color='black', s=100, label="Ponto B", zorder=5)

ax.set_xlabel("LONGITUDE")
ax.set_ylabel("LATITUDE")
ax.set_title("Trajetórias Ida e Volta (dinâmico)")
ax.legend(loc='lower left')

plt.tight_layout()
plt.show()
