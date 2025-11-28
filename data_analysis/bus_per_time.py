import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# === Carregar todos os arquivos RData ===
arquivos = glob.glob("../data/bus_csv/sul/LINHA_*_COMPLETO.csv")  # pega todos os arquivos no padrão

dfs = []
for arquivo in arquivos:
    df = pd.read_csv(arquivo)

    df = df[["GPSTIMESTAMP", "LINE", "VELOCITY", "PARKING", 'BUSID']].dropna(subset=["GPSTIMESTAMP", "LINE", "VELOCITY"])
    df = df[df["PARKING"].isna()]
    #df = df[df["VELOCITY"] > 0]

    # Converte timestamp
    df["GPSTIMESTAMP"] = pd.to_datetime(df["GPSTIMESTAMP"], errors="coerce")

    # Extrair hora
    df["HORA"] = df["GPSTIMESTAMP"].dt.hour  # hora inteira (0–23)

    dfs.append(df)

# Junta tudo em um único DataFrame
dados = pd.concat(dfs, ignore_index=True)

# Garante que o timestamp está como datetime
dados["GPSTIMESTAMP"] = pd.to_datetime(dados["GPSTIMESTAMP"], errors="coerce")

# Cria uma coluna com a hora (0–23)
dados["HOUR"] = dados["GPSTIMESTAMP"].dt.hour

# Conta quantos ônibus distintos (busid) aparecem em cada hora
onibus_por_hora = dados.groupby("HOUR")["BUSID"].nunique().reset_index()

onibus_por_hora.columns = ["Hora", "Onibus_em_circulacao"]

print(onibus_por_hora)


plt.figure(figsize=(10,6))
sns.lineplot(data=onibus_por_hora, x="Hora", y="Onibus_em_circulacao", marker="o", color="#E63946")
plt.title("Quantidade de ônibus em circulação por hora do dia")
plt.xlabel("Hora do dia")
plt.ylabel("Nº de ônibus em circulação")
plt.xticks(range(0, 24, 1))  # marca cada hora
plt.xlim(0, 23)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

onibus_por_linha_hora = dados.groupby(["LINE", "HOUR"])["BUSID"].nunique().reset_index()
onibus_por_linha_hora.columns = ["Linha", "Hora", "Onibus_em_circulacao"]

# Plot 2 (por linha)
plt.figure(figsize=(10,6))
sns.lineplot(data=onibus_por_linha_hora, x="Hora", y="Onibus_em_circulacao", hue="Linha", marker="o")
plt.title("Ônibus em circulação por linha e hora do dia")
plt.xlabel("Hora do dia")
plt.ylabel("Nº de ônibus em circulação")
plt.xticks(range(0, 24, 1))  # marca cada hora
plt.xlim(0, 23)
plt.tight_layout()
plt.show()