import yaml
import argparse
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def PriceMerge(row, data, key):
    value = np.zeros(len(key))
    cols = np.array(data.columns)[1:]
    label = np.array(row)[1:]
    dt = pd.DataFrame([value], columns=key, dtype=int)
    for i, col in enumerate(cols):
        for k in key:
            if k in col:
                if (label[i] > 0):
                    dt[k] = 1

    return dt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract Aspect from Data')
    parser.add_argument('--domain', '--dm', type=str, required=True,
                        help='Tên domain', choices=['restaurant', 'hotel'])

    parser.add_argument('--languague', '--lang', type=str,
                        required=True, help='Ngôn ngữ', choices=['en', 'dutch'])

    parser.add_argument('--type', '--tp', type=str,
                        required=True, help='Train/Dev/Test', choices=['train', 'dev', 'test'])

    parser.add_argument('--check', '--ck', type=bool,
                        help='Check data', default=False)

    args = parser.parse_args()

    with open('semeval/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    domain = args.domain
    lang = args.languague
    check = args.check
    typeCSV = args.type + '_csv'
    typeAspect = typeCSV + '_aspects'
    typeLabel = typeAspect + '_relabel'

    if (domain == 'restaurant'):
        path = config[domain][lang][typeAspect]
        pathLabel = config[domain][lang][typeLabel]
        if not check:
            data = pd.read_csv(path)

            key = ['Review', 'AMBIENCE', 'QUALITY',
                   'PRICES', 'LOCATION', 'SERVICE']
            newCol = data.apply(lambda row: PriceMerge(
                row, data, key[1:]), axis=1)
            result_df = pd.concat(
                [data, pd.concat(newCol.values).reset_index(drop=True)], axis=1)
            result_df = result_df[key]
            print('==========================================================')
            print('Result DF', result_df.info())

            print('==========================================================')

            last_df = result_df[(result_df[key[1:]] != 0).any(axis=1)]
            print('Last DF', last_df.info())
            print('==========================================================')

            last_df.to_csv(pathLabel, index=False)
        else:
            data = pd.read_csv(pathLabel)
            filtered_df = data[data['LOCATION'] == 1]
            print('==========================================================')
            print('LOCATION : ', data['LOCATION'].sum())
            print('SERVICE : ', data['SERVICE'].sum())
            print('AMBIENCE : ', data['AMBIENCE'].sum())
            print('QUALITY : ', data['QUALITY'].sum())
            print('PRICES : ', data['PRICES'].sum())
            print('==========================================================')
            print('LOCATION : ', filtered_df['LOCATION'].sum())
            print('SERVICE : ', filtered_df['SERVICE'].sum())
            print('AMBIENCE : ', filtered_df['AMBIENCE'].sum())
            print('QUALITY : ', filtered_df['QUALITY'].sum())
            print('PRICES : ', filtered_df['PRICES'].sum())
