import pandas as pd
df = pd.read_csv('ML Project/data/raw/dataset siralatriste.csv')

#df = pd.read_csv('data/raw/dataset siralatriste.csv')


df.to_csv('processedpy.csv', index=False)