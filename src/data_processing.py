import pandas as pd
data = pd.read_csv('ML Project/data/raw/dataset siralatriste.csv')

#para sacar la media de compartidos de cada uno de mis juegos 
data['Med Juego Compartido'] = data.groupby('Juego')['Compartido'].transform('mean')

#para sacar la media de comentarios de cada uno de mis juegos 
data['Med Juego Comentarios'] = data.groupby('Juego')['Comentarios'].transform('mean')

data.to_csv('ML Project/data/processed.csv', index=False)