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
# ---- 1) Calcular velocidade média e contagem por hora ----
vel_por_hora = dados.groupby(["LINE", "HORA"]).agg(
    media_vel=("VELOCITY", "mean"),
    count=("VELOCITY", "count")
).reset_index()

# Cores
colors = ["#E63946", "#2A9D8F", "#457B9D", "#F4A261", "#1D3557"]

# ---- 2) Filtrar horas com dados suficientes ----
min_registros = 100
horas_validas = (
    vel_por_hora.groupby("HORA")["count"].sum()
    .loc[lambda x: x >= min_registros].index
)

vel_por_hora_filtrado = vel_por_hora[vel_por_hora["HORA"].isin(horas_validas)]

plt.figure(figsize=(12,7))

sns.lineplot(
    data=vel_por_hora_filtrado,
    x="HORA",
    y="media_vel",
    hue="LINE",
    marker="o",
    palette=colors
)

plt.title("Velocidade média por hora (apenas horas com dados suficientes)")
plt.xlabel("Hora do dia")
plt.ylabel("Velocidade média (km/h)")
plt.legend(title="Linha")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,7))

contagem_por_hora = vel_por_hora.groupby("HORA")["count"].sum().reset_index()

sns.barplot(
    data=contagem_por_hora,
    x="HORA",
    y="count",
    color="gray"
)

plt.title("Quantidade de registros por hora (todas as linhas)")
plt.xlabel("Hora do dia")
plt.ylabel("Número de registros")
plt.tight_layout()
plt.show()
