import pandas as pd
import numpy as np

df = pd.read_parquet('parquets/G1-2023-02-21-A.parquet')
unique_lines = df['LINE'].unique()
randNum = np.random.choice(unique_lines.shape[0], size=10, replace=False)
sample_lines_str = ', '.join(unique_lines[randNum].astype(str))

line = input(f'Escolha uma linha\n' +
             'Exemplos de opções:\n' + (sample_lines_str) + '\n')

df = df[(df['LINE'] == line)]

unique_ids = df['BUSID'].unique()
sample_size = min(10, unique_ids.shape[0])
randNum = np.random.choice(unique_ids.shape[0], size=sample_size, replace=False)
sample_ids_str = ', '.join(unique_ids[randNum].astype(str))

busid = input(f'Escolha um id:\n' +
             'Exemplos de opções:\n' + (sample_ids_str) + '\n')

df = df[(df['BUSID'] == busid)]

if input('Gerar csv? S/N: ').strip().upper() == 'S':
    df.to_csv('data/bus_csv/LINHA_' + line + '_' + busid + '_COMPLETO.csv', index = False)

print('CSV criado!')