import pyreadr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# === Carregar todos os arquivos RData ===
arquivos = glob.glob("linha_*_ida_volta.RData")  # pega todos os arquivos no padrão

dfs = []
for arquivo in arquivos:
    result = pyreadr.read_r(arquivo)
    df = list(result.values())[0]

    # Seleciona colunas importantes
    df = df[["GPSTIMESTAMP", "LINE", "VELOCITY"]].dropna()

    # Converte timestamp
    df["GPSTIMESTAMP"] = pd.to_datetime(df["GPSTIMESTAMP"], errors="coerce")

    # Extrair hora e hora inteira já no df
    df["HORA"] = df["GPSTIMESTAMP"].dt.time        # pega só HH:MM:SS

    # Filtra velocidades > 0
    df = df[df["VELOCITY"] > 0]

    # Adiciona info da linha a partir do próprio arquivo
    df["LINE_FILE"] = arquivo.split("_")[1]  # extrai o número da linha do nome do arquivo

    dfs.append(df)

# Junta tudo em um único DataFrame
dados = pd.concat(dfs, ignore_index=True)

print("Dados carregados:", dados["LINE_FILE"].unique())

#--------------------------------------
# Velocidade média por hora do dia
vel_por_hora = dados.groupby(
    ["LINE_FILE", dados["GPSTIMESTAMP"].dt.hour]
)["VELOCITY"].mean().reset_index()

# Renomear a coluna da hora para algo mais claro
vel_por_hora.rename(columns={"GPSTIMESTAMP": "HOUR"}, inplace=True)

# === Plot 1: Velocidade ao longo do tempo por linha ===
plt.figure(figsize=(10,6))
sns.lineplot(data=vel_por_hora, x="HOUR", y="VELOCITY", hue="LINE_FILE", marker="o")
plt.title("Velocidade média por hora do dia")
plt.xlabel("Hora do dia")
plt.ylabel("Velocidade média (km/h)")
plt.legend(title="Linha")
plt.tight_layout()
plt.show()

# === Plot 2: Distribuição da velocidade por linha (Boxplot) ===
plt.figure(figsize=(8,6))
sns.boxplot(data=dados, x="LINE_FILE", y="VELOCITY")
plt.title("Distribuição da velocidade por linha de ônibus")
plt.xlabel("Linha")
plt.ylabel("Velocidade (km/h)")
plt.tight_layout()
plt.show()

# === Plot 3: Histograma de velocidades por linha ===
plt.figure(figsize=(10,6))
sns.histplot(data=dados, x="VELOCITY", hue="LINE_FILE", kde=True, element="step", common_norm=False)
plt.title("Distribuição de frequências de velocidade por linha")
plt.xlabel("Velocidade (km/h)")
plt.ylabel("Frequência")
plt.tight_layout()
plt.show()
