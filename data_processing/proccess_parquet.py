import pandas as pd
import numpy as np

# Lendo os Parquets
df_A = pd.read_parquet('data/parquets/G1-2023-02-21-A.parquet')
df_B = pd.read_parquet('data/parquets/G1-2023-02-21-B.parquet')

# Fazendo merge pelo ID
df = df_A.merge(df_B, on='ID', how='left')  # 'left' mantém todos os registros de A

unique_lines = df['LINE'].unique()
print(unique_lines)
randNum = np.random.choice(unique_lines.shape[0], size=10, replace=False)
sample_lines_str = ', '.join(unique_lines[randNum].astype(str))

line = input(f'Escolha uma linha\n' +
             'Exemplos de opções:\n' + (sample_lines_str) + '\n')

df = df[(df['LINE'] == line)]

# unique_ids = df['BUSID'].unique()
# sample_size = min(10, unique_ids.shape[0])
# randNum = np.random.choice(unique_ids.shape[0], size=sample_size, replace=False)
# sample_ids_str = ', '.join(unique_ids[randNum].astype(str))

# busid = input(f'Escolha um id:\n' +
#              'Exemplos de opções:\n' + (sample_ids_str) + '\n')

# df = df[(df['BUSID'] == busid)]

if df.empty: 
    print('Linha não encontrada!')
    
elif input('Gerar csv? S/N: ').strip().upper() == 'S':
  
    df.to_csv('data/bus_csv/LINHA_' + line + '_COMPLETO.csv', index = False)
    print('CSV criado!')