from pathlib import Path
import sqlite3
import pandas as pd
# create/connect do db
Path('my_data.db').touch()
conn = sqlite3.connect('my_data.db')
c = conn.cursor()
# data load
spy = pd.read_csv('Datos/SPY.csv')
#load data to 
spy.to_sql('spy', conn, if_exists='replace', index=False)

#close connection
conn.close()