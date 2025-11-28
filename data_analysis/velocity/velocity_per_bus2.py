import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import numpy as np
import matplotlib.dates as mdates
import pytz   # pip install pytz se não tiver

# ---- CONFIG ----
arquivos = glob.glob("../../data/bus_csv/sul/LINHA_*_COMPLETO.csv")
PLOTS_MAX_BUS = 12        # quantidade máxima de BUSIDs a plotar
MIN_VEL_KMH = 0.5
MAX_VEL_KMH = 140.0
ASSUME_TIMESTAMP_TZ = "UTC"        # se seus timestamps forem UTC, deixe "UTC". 
# Se já estiverem no fuso local e sem tz, colocar None.
TARGET_TZ = "America/Sao_Paulo"    # fuso que você quer ver nos gráficos (ajuste se necessário)

SHOW_RAW_POINTS = True     # mostrar bolinhas (dados brutos)
SHOW_ROLLING = True        # mostrar linha suavizada
ROLLING_WINDOW = "5min"    # janela da média móvel (opções: "5min", "10min", etc.)
ONLY_PLOT_LINE_IF_ENOUGH = True  # só plotar BUSIDs com >= N pontos (abaixo evita "ruído")
MIN_POINTS_TO_PLOT = 30

sns.set_style("whitegrid")

dfs = []
for arquivo in arquivos:
    df = pd.read_csv(arquivo)

    # checa colunas mínimas
    need_cols = {"GPSTIMESTAMP", "LINE", "BUSID", "VELOCITY"}
    if not need_cols.issubset(set(df.columns)):
        print(f"[AVISO] Arquivo {arquivo} não tem colunas esperadas, pulando.")
        continue

    # parse timestamp (robusto)
    df["GPSTIMESTAMP"] = pd.to_datetime(df["GPSTIMESTAMP"], errors="coerce", infer_datetime_format=True)
    df = df.dropna(subset=["GPSTIMESTAMP"])

    # Se você sabe que os timestamps são UTC mas aparecem sem tz (naive),
    # podemos localizá-los e depois converter para o fuso alvo:
    if ASSUME_TIMESTAMP_TZ is not None:
        # se o index já tiver tz info, não localiza duas vezes
        if df["GPSTIMESTAMP"].dt.tz is None:
            df["GPSTIMESTAMP"] = df["GPSTIMESTAMP"].dt.tz_localize(ASSUME_TIMESTAMP_TZ)
        # converte para fuso alvo (ajusta horário para exibir local)
        df["GPSTIMESTAMP"] = df["GPSTIMESTAMP"].dt.tz_convert(TARGET_TZ)
    else:
        # se ASSUME_TIMESTAMP_TZ is None, mantemos timestamps como estão (naive)
        pass

    # ordena por tempo
    df = df.sort_values("GPSTIMESTAMP").reset_index(drop=True)

    # PARKING (se existir) mantemos NaN como antes
    if "PARKING" in df.columns:
        df = df[df["PARKING"].isna()]

    # VELOCITY numérico e conversão m/s -> km/h (heurística)
    df["VELOCITY"] = pd.to_numeric(df["VELOCITY"], errors="coerce")
    df = df.dropna(subset=["VELOCITY"])
    med = df["VELOCITY"].median()
    vmax = df["VELOCITY"].max()
    if (med < 6 and vmax <= 30) or (vmax <= 20 and med < 4):
        df["VELOCITY"] = df["VELOCITY"] * 3.6
        unidade = "m/s -> km/h (convertido)"
    else:
        unidade = "km/h (assumido)"

    # filtra zeros e outliers
    df = df[df["VELOCITY"].between(MIN_VEL_KMH, MAX_VEL_KMH)]

    dfs.append(df)

if len(dfs) == 0:
    raise SystemExit("Nenhum arquivo válido carregado.")

dados = pd.concat(dfs, ignore_index=True)

# Map BUSID -> LINE (mostra para você)
map_bus_line = dados.groupby("BUSID")["LINE"].agg(lambda x: x.mode().iloc[0] if len(x) else None)
print("Mapa BUSID -> LINE (exemplo dos primeiros 20):")
print(map_bus_line.head(20))

# Filtra BUSIDs para plot (limitado)
bus_ids = list(map_bus_line.index)
if len(bus_ids) > PLOTS_MAX_BUS:
    print(f"[INFO] Há {len(bus_ids)} BUSIDs; serão plotados os primeiros {PLOTS_MAX_BUS}.")
    bus_ids = bus_ids[:PLOTS_MAX_BUS]

# Função helper para formatar eixo de datas
def format_time_axis(ax, tz_aware=True):
    # usa locator por hora e rotaciona labels
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0,24,2)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M", tz=pytz.timezone(TARGET_TZ) if ASSUME_TIMESTAMP_TZ else None))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)

# Plot por BUSID
for bus in bus_ids:
    df_bus = dados[dados["BUSID"] == bus].sort_values("GPSTIMESTAMP")
    if df_bus.shape[0] < MIN_POINTS_TO_PLOT and ONLY_PLOT_LINE_IF_ENOUGH:
        print(f"[INFO] Pulando {bus} (apenas {df_bus.shape[0]} pontos).")
        continue

    fig, ax = plt.subplots(figsize=(12,4))
    # pontos brutos
    if SHOW_RAW_POINTS:
        ax.scatter(df_bus["GPSTIMESTAMP"].dt.tz_convert(TARGET_TZ) if df_bus["GPSTIMESTAMP"].dt.tz is not None else df_bus["GPSTIMESTAMP"],
                   df_bus["VELOCITY"], s=10, alpha=0.5, label="dados brutos")

    # rolling (média móvel) — precisa de índice datetime
    if SHOW_ROLLING:
        # cria série com index datetime (necessário tz-aware)
        ser = df_bus.set_index("GPSTIMESTAMP")["VELOCITY"]
        # se a série for tz-aware, rolling("5min") funciona; caso seja naive, convertemos para timezone alvo sem alterar os valores
        try:
            roll = ser.rolling(ROLLING_WINDOW, min_periods=1).mean()
            ax.plot(roll.index, roll.values, color="#e63946", linewidth=1.6, label=f"rolling {ROLLING_WINDOW}")
        except Exception as e:
            # fallback: reindex por segundos e usar rolling por número de pontos
            roll = ser.rolling(window=5, min_periods=1).mean()
            ax.plot(roll.index, roll.values, color="#e63946", linewidth=1.6, label=f"rolling fallback (pontos)")

    # título incluindo LINE
    linha = map_bus_line.loc[bus]
    ax.set_title(f"Ônibus {bus} — Linha {linha} — Velocidade ({unidade})")
    ax.set_xlabel("Tempo")
    ax.set_ylabel("Velocidade (km/h)")
    ax.legend()

    # formatar eixo x
    format_time_axis(ax)

    plt.tight_layout()
    plt.show()
