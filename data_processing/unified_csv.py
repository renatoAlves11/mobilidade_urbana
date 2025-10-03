import pandas as pd
import glob

# Todos os CSVs na pasta
csv_files = glob.glob("*.csv")

# Concatenar direto usando uma list comprehension
df_final = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

# Salvar em um CSV unificado
df_final.to_csv("arquivo_unificado.csv", index=False)
