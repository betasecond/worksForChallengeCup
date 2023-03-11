import pandas as pd
read_file = pd.read_excel(r'census.xlsx', sheet_name='Sheet2')
read_file.to_csv(r'data.csv', index=None, header=True)