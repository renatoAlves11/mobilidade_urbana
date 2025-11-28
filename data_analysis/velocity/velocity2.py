import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# === Carregar todos os arquivos CSV ===
arquivos = glob.glob("../../data/bus_csv/sul/LINHA_*_COMPLETO.csv")

dfs = []
dfs1 = []

for arquivo in arquivos:
    df = pd.read_csv(arquivo)

    # Selecionar colunas essenciais + remover linhas sem timestamp/line/velocity
    df = df[["GPSTIMESTAMP", "LINE", "VELOCITY", "PARKING"]].dropna(
        subset=["GPSTIMESTAMP", "LINE", "VELOCITY"]
    )

    # Remover registros em que o ônibus está estacionado
    df = df[df["PARKING"].isna()]

    # Converter timestamp
    df["GPSTIMESTAMP"] = pd.to_datetime(df["GPSTIMESTAMP"], errors="coerce")

    # Extrair hora
    df["HORA"] = df["GPSTIMESTAMP"].dt.hour

    # === 📌 CONVERTER m/s → km/h ===
    df["VELOCITY"] = df["VELOCITY"] * 3.6

    # === 📌 Remover outliers absurdos (> 80 km/h) ===
    df = df[df["VELOCITY"].between(0.1, 80)]

    # df1 (para histogramas, sem zeros)
    df1 = df[df["VELOCITY"] > 0.5]

    dfs.append(df)
    dfs1.append(df1)

# Junta tudo
dados = pd.concat(dfs, ignore_index=True)
dados1 = pd.concat(dfs1, ignore_index=True)

print("Linhas carregadas:", dados["LINE"].unique())

#--------------------------------------
# Velocidade média por hora do dia
vel_por_hora = dados.groupby(["LINE", "HORA"])["VELOCITY"].mean().reset_index()

# Cores
colors = ["#E63946", "#2A9D8F", "#457B9D", "#F4A261", "#1D3557"]

# === Plot 1A: Velocidade até meio-dia ===
plt.figure(figsize=(10,6))
sns.lineplot(
    data=vel_por_hora, x="HORA", y="VELOCITY",
    hue="LINE", marker="o", palette=colors
)
plt.title("Velocidade média por hora do dia (0h – 23h)")
plt.xlabel("Hora do dia")
plt.ylabel("Velocidade média (km/h)")
plt.legend(title="Linha")
plt.tight_layout()
plt.show()

# === Plot 2: Boxplot de velocidade por linha ===
plt.figure(figsize=(8,6))
sns.boxplot(data=dados, x="LINE", y="VELOCITY", palette=colors)
plt.title("Distribuição da velocidade por linha de ônibus")
plt.xlabel("Linha")
plt.ylabel("Velocidade (km/h)")
plt.tight_layout()
plt.show()

# === Plot 3: Histograma ===
plt.figure(figsize=(10,6))
sns.histplot(
    data=dados1, x="VELOCITY",
    hue="LINE", kde=True, element="step",
    common_norm=False, palette=colors
)
plt.title("Distribuição de frequências de velocidade por linha")
plt.xlabel("Velocidade (km/h)")
plt.ylabel("Frequência")
plt.tight_layout()
plt.show()
