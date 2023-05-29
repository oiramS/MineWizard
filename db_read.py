from pathlib import Path
import sqlite3
import pandas as pd
# create/connect do db
Path('my_data.db').touch()
conn = sqlite3.connect('my_data.db')
c = conn.cursor()
# query the db
spy = pd.read_sql_query('''select * from spy''', conn)
#show data
print(spy)
#close connection
conn.close()