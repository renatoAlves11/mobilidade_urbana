import pandas as pd
import matplotlib.pyplot as plt
import os
 
# Files provided
paths = [
    "../../data/bus_csv/sul/LINHA_432_COMPLETO.csv",
    "../../data/bus_csv/sul/LINHA_455_COMPLETO.csv",
    "../../data/bus_csv/sul/LINHA_457_COMPLETO.csv",
    "../../data/bus_csv/sul/LINHA_472_COMPLETO.csv",
    "../../data/bus_csv/sul/LINHA_484_COMPLETO.csv"
]
 
# Function to load, extract hour and plot histogram stylized
def plot_hour_hist(path):
    df = pd.read_csv(path)
 
    if "GPSTIMESTAMP" in df.columns:
        df["GPSTIMESTAMP"] = pd.to_datetime(df["GPSTIMESTAMP"], errors="coerce")
        df = df.dropna(subset=["GPSTIMESTAMP"])
        df["HOUR"] = df["GPSTIMESTAMP"].dt.hour
 
        # Determine hour with most records
        hour_counts = df["HOUR"].value_counts().sort_index()
        peak_hour = hour_counts.idxmax()
 
        # Plot
        plt.figure(figsize=(9, 5))
        plt.hist(
            df["HOUR"],
            bins=24,
            color="skyblue",
            edgecolor="black"
        )
 
        # Line marking the hour with most records
        plt.axvline(
            x=peak_hour,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Hora mais frequente: {peak_hour}h"
        )
 
        plt.title(f"Histograma de registros por hora — {os.path.basename(path)}")
        plt.xlabel("Hora do dia")
        plt.ylabel("Quantidade de registros")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.4)
 
        plt.show()
    else:
        print(f"Arquivo {path} não possui coluna GPSTIMESTAMP.")
 
 
# Plot histograms for each file
for p in paths:
    plot_hour_hist(p)