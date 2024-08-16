import pandas as pd
import glob

# python sentiment/remove_duplicate_data.py
if __name__ == '__main__':
    files = glob.glob('sentiment/relabel/*/*/*.csv')
    print(files)
    for file in files:
        df = pd.read_csv(file)
        print('Before :', df.info())
        df = df.drop_duplicates(subset=['Review'], keep='first')
        print('After :', df.info())
        df.to_csv(file, index=False)
