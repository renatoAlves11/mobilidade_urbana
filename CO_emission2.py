import pyreadr
import pandas as pd
import folium
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

# Exemplo de mapeamento de códigos numéricos para bairros
bairro_map = {
    "002": "Gamboa",
    "003": "Santo Cristo",
    "004": "Caju",
    "005": "Centro",
    "006": "Catumbi",
    "007": "Rio Comprido",
    "008": "Cidade Nova",
    "009": "Estácio",
    "010": "Imperial de São Cristóvão",
    "012": "Benfica",
    "014": "Santa Teresa",
    "015": "Flamengo",
    "016": "Glória",
    "017": "Laranjeiras",
    "020": "Botafogo",
    "022": "Urca",
    "024": "Copacabana",
    "032": "Praça da Bandeira",
    "033": "Tijuca",
    "035": "Maracanã",
    "036": "Vila Isabel",
    "038": "Grajaú",
    "039": "Manguinhos",
    "040": "Bonsucesso",
    "050": "Higienópolis",
    "053": "Del Castilho",
    "054": "Inhaúma",
    "057": "São Francisco Xavier",
    "058": "Rocha",
    "059": "Riachuelo",
    "060": "Sampaio",
    "061": "Engenho Novo",
    "062": "Lins de Vasconcelos",
    "063": "Méier",
    "066": "Engenho de Dentro",
    "067": "Água Santa",
    "068": "Encantado",
    "070": "Abolição",
    "071": "Pilares",
    "093": "Cacuia",
    "096": "Cocotá",
    "097": "Bancários",
    "099": "Jardim Guanabara",
    "101": "Tauá",
    "102": "Moneró",
    "103": "Portuguesa",
    "104": "Galeão",
    "105": "Cidade Universitária",
    "115": "Jacarepaguá",
    "116": "Anil",
    "117": "Gardênia Azul",
    "118": "Cidade de Deus",
    "119": "Curicica",
    "120": "Freguesia (Jacarepaguá)",
    "121": "Pechincha",
    "122": "Taquara",
    "123": "Tanque",
    "127": "Itanhangá",
    "128": "Barra da Tijuca",
    "129": "Camorim",
    "157": "Maré",
    "158": "Vasco da Gama"
}

regiao_administrativa_map = {
    1: "Portuária",
    2: "Centro",
    3: "Rio Comprido",
    4: "Botafogo",
    5: "Copacabana",
    6: "Lagoa",
    7: "São Cristóvão",
    8: "Tijuca",
    9: "Vila Isabel",
    10: "Ramos",
    11: "Penha",
    12: "Inhaúma",
    13: "Méier",
    14: "Irajá",
    15: "Madureira",
    16: "Jacarepaguá",
    17: "Bangu",
    18: "Campo Grande",
    19: "Santa Cruz",
    20: "Ilha do Governador",
    21: "Ilha de Paquetá",
    22: "Anchieta",
    23: "Santa Teresa",
    24: "Barra da Tijuca",
    25: "Pavuna",
    26: "Guaratiba",
    27: "Rocinha",
    28: "Jacarezinho",
    29: "Complexo do Alemão",
    30: "Maré",
    31: "Vigário Geral",
    33: "Realengo",
    34: "Cidade de Deus"
}

# Substituindo os códigos numéricos pelos nomes dos bairros
dados["NEIGHBORHOOD"] = dados["NEIGHBORHOOD"].map(bairro_map)

# Substituindo os códigos numéricos pelos nomes das regiões administrativas
dados["ADMINISTRATIVEREGION"] = dados["ADMINISTRATIVEREGION"].map(regiao_administrativa_map)

#--------------------------------------
# 1- CO2 médio por bairro
co2_por_bairro = dados.groupby("NEIGHBORHOOD")["CO_2"].mean().sort_values(ascending=False).reset_index()
plt.figure(figsize=(12,6))
sns.barplot(data=co2_por_bairro, x="CO_2", y="NEIGHBORHOOD", palette="viridis")
plt.title("CO₂ médio por bairro")
plt.xlabel("CO₂ médio (ppm)")
plt.ylabel("Bairro")
plt.tight_layout()
plt.show()

#--------------------------------------
# 2- CO2 médio por região administrativa
co2_por_regiao = dados.groupby("ADMINISTRATIVEREGION")["CO_2"].mean().sort_values(ascending=False).reset_index()
plt.figure(figsize=(10,6))
sns.barplot(data=co2_por_regiao, x="ADMINISTRATIVEREGION", y="CO_2", palette="magma")
plt.title("CO₂ médio por região administrativa")
plt.xlabel("Região")
plt.ylabel("CO₂ médio (ppm)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#--------------------------------------
# 3- CO2 no mapa (latitude x longitude)
# Centro do RJ para centralizar
mapa = folium.Map(location=[-22.9068, -43.1729], zoom_start=11)

# Adicionar pontos
for _, row in dados.iterrows():
    folium.CircleMarker(
        location=[row['LATITUDE'], row['LONGITUDE']],
        radius=row['CO_2']/10,  # ajustar tamanho
        color='red',
        fill=True,
        fill_opacity=0.6,
        popup=f"CO₂: {row['CO_2']}"
    ).add_to(mapa)

mapa.save("mapa_CO2.html")

#--------------------------------------
# 4- CO2 por hora do dia (linha + bairro)
co2_por_hora_bairro = dados.groupby(["HORA", "NEIGHBORHOOD"])["CO_2"].mean().reset_index()
plt.figure(figsize=(12,6))
sns.lineplot(data=co2_por_hora_bairro, x="HORA", y="CO_2", hue="NEIGHBORHOOD", marker="o")
plt.title("CO₂ médio por hora do dia e bairro")
plt.xlabel("Hora do dia")
plt.ylabel("CO₂ médio (ppm)")
plt.legend(title="Bairro", bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout()
plt.show()
