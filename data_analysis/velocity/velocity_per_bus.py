import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# === Carregar todos os arquivos CSV ===
arquivos = glob.glob("../../data/bus_csv/sul/LINHA_*_COMPLETO.csv")

dfs = []
for arquivo in arquivos:
    df = pd.read_csv(arquivo)

    df = df[["GPSTIMESTAMP", "LINE", "BUSID", "VELOCITY", "PARKING"]].dropna(subset=["GPSTIMESTAMP", "LINE", "VELOCITY"])
    df = df[df["PARKING"].isna()]
    
    df["GPSTIMESTAMP"] = pd.to_datetime(df["GPSTIMESTAMP"], errors="coerce")
    dfs.append(df)

# Junta tudo
dados = pd.concat(dfs, ignore_index=True)

# --- Mostra quantos ônibus há ---
bus_ids = dados["BUSID"].unique()
print(f"Total de ônibus encontrados: {len(bus_ids)}")
print("Exemplo de BUSID:", bus_ids[:5])

# --- Cria gráfico individual para cada BUSID ---
for bus in bus_ids:
    df_bus = dados[dados["BUSID"] == bus]

    plt.figure(figsize=(10,5))
    sns.scatterplot(
        data=df_bus,
        x="GPSTIMESTAMP",
        y="VELOCITY",
        s=15,               # tamanho dos pontos
        color="#1D3557",
        alpha=0.7
    )
    plt.title(f"Variação da velocidade ao longo do tempo — Ônibus {bus}")
    plt.xlabel("Tempo")
    plt.ylabel("Velocidade (km/h)")
    plt.tight_layout()
    plt.show()