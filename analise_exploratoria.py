import pandas as pd
import pyreadr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ==========================
# DICIONÁRIO DE PONTOS POR LINHA
# ==========================
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

# raw_232 = "https://raw.githubusercontent.com/cefet-rj-temas/grupo2/refs/heads/main/route-data/results/resultado_segmentacao_LINHA_232_B25542_COMPLETO.csv?token=GHSAT0AAAAAADKCXGRFYYMV2IZSYGI5E4SY2GIFMUQ"
# df_232 = pd.read_csv(raw_232)

result = pyreadr.read_r("linha_232.RData")
result1 = pyreadr.read_r("linha_232_ida_volta.RData")

print(result1.keys())
df = result1['filtered_data']
print(df.columns)

A = pontos_por_linha['232']['A']
B = pontos_por_linha['232']['B']

# Plot 1
plt.figure(figsize=(12, 6))
plt.scatter(df['LONGITUDE'], df['LATITUDE'], c=['green'], s=10)
plt.plot(df['LONGITUDE'], df['LATITUDE'], linestyle='--', alpha=0.5)
plt.scatter(*A, color='black', label='Ponto A (Chegada)', zorder=5)
plt.scatter(*B, color='orange', label='Ponto B (Saída)', zorder=5)
#plt.title(f"Direção Instantânea (Produto Escalar) - {os.path.splitext(os.path.basename(filepath))[0]}")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.axis('equal')
plt.legend(handles=[
    mpatches.Patch(color='green', label='Ida A → B'),
    mpatches.Patch(color='red', label='Volta B → A'),
    mpatches.Patch(color='blue', label='Parado/Indefinido'),
    plt.Line2D([], [], marker='o', color='w', label='Ponto A (Chegada)', markerfacecolor='black', markersize=8),
    plt.Line2D([], [], marker='o', color='w', label='Ponto B (Saída)', markerfacecolor='orange', markersize=8)
])
plt.show()