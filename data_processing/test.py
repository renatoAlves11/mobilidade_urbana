import pandas as pd
import numpy as np

df = pd.read_parquet('data/parquets/G1-2023-02-21-B.parquet')
print(df.loc[:, "PARKING"])