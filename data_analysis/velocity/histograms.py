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

# Function to load, extract hour and plot histogram
def plot_hour_hist(path):
    df = pd.read_csv(path)
    
    # Convert timestamp
    if "GPSTIMESTAMP" in df.columns:
        df["GPSTIMESTAMP"] = pd.to_datetime(df["GPSTIMESTAMP"], errors="coerce")
        df = df.dropna(subset=["GPSTIMESTAMP"])
        df["HOUR"] = df["GPSTIMESTAMP"].dt.hour
        
        plt.figure(figsize=(8,5))
        plt.hist(df["HOUR"], bins=24)
        plt.title(f"Histograma de quantidade de registros por hora\n{os.path.basename(path)}")
        plt.xlabel("Hora do dia")
        plt.ylabel("Quantidade de registros")
        plt.show()
    else:
        print(f"Arquivo {path} não possui coluna GPSTIMESTAMP.")

# Plot histograms for each file
for p in paths:
    plot_hour_hist(p)
