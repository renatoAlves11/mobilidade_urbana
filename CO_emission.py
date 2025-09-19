import pyreadr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# === Carregar todos os arquivos RData ===
arquivos = glob.glob("linha_*_ida_volta.RData")

dfs = []
for arquivo in arquivos:
    result = pyreadr.read_r(arquivo)
    df = list(result.values())[0]

    # Seleciona colunas importantes
    df = df[["GPSTIMESTAMP", "LINE", "CO_2", "VELOCITY"]].dropna()  # CO2 + VELOCITY para correlação

    # Converte timestamp
    df["GPSTIMESTAMP"] = pd.to_datetime(df["GPSTIMESTAMP"], errors="coerce")

    # Extrair hora
    df["HORA"] = df["GPSTIMESTAMP"].dt.hour

    # Filtra valores inválidos
    df = df[df["CO_2"] > 0]

    # Adiciona info da linha a partir do próprio arquivo
    df["LINE_FILE"] = arquivo.split("_")[1]

    dfs.append(df)

# Junta tudo em um único DataFrame
dados = pd.concat(dfs, ignore_index=True)

print("Dados carregados:", dados["LINE_FILE"].unique())

#--------------------------------------
# CO2 médio por hora do dia
co2_por_hora = dados.groupby(
    ["LINE_FILE", "HORA"]
)["CO_2"].mean().reset_index()

# === Plot 1: CO2 médio por hora ===
plt.figure(figsize=(12,6))
sns.lineplot(
    data=co2_por_hora, 
    x="HORA", 
    y="CO_2", 
    hue="LINE_FILE", 
    marker="o",
    palette="tab10",     # paleta de cores mais limpa
    alpha=0.8
)

plt.title("CO₂ médio por hora do dia", fontsize=16)
plt.xlabel("Hora do dia", fontsize=12)
plt.ylabel("CO₂ médio (ppm)", fontsize=12)
plt.xticks(range(0,24))  # mostrar todas as horas
plt.grid(alpha=0.3)
plt.legend(title="Linha", bbox_to_anchor=(1.05, 1), loc='upper left')  # legenda fora do gráfico
plt.tight_layout()
plt.show()

# === Plot 2: Histograma de CO2 por linha ===
plt.figure(figsize=(10,6))
sns.histplot(data=dados, x="CO_2", hue="LINE_FILE", kde=True, element="step", common_norm=False)
plt.title("Distribuição de frequências de CO₂ por linha")
plt.xlabel("CO₂ (ppm)")
plt.ylabel("Frequência")
plt.tight_layout()
plt.show()

# === Plot 3: Correlação CO2 x Velocidade ===
plt.figure(figsize=(8,6))
sns.scatterplot(data=dados, x="VELOCITY", y="CO_2", hue="LINE_FILE", alpha=0.6)
plt.title("Correlação entre velocidade e CO₂")
plt.xlabel("Velocidade (km/h)")
plt.ylabel("CO₂ (ppm)")
plt.tight_layout()
plt.show()
