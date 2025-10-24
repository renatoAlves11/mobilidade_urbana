import pandas as pd
import numpy as np

parqs = ['A','B','C','D','E']

for i in range(len(parqs)):
    print('Parquet: ' + parqs[i])
    df = pd.read_parquet('parquets/G1-2023-02-21-' + parqs[i] + '.parquet')

    print(df.columns)
    print(df)
    print('\n')