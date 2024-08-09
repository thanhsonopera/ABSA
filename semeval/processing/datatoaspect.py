import yaml
import argparse
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def get_label_restaurant(data, label):
    target = []
    for l in label:
        if l == 'DRINKS#STYLE&OPTIONS':
            l = 'DRINKS#STYLE_OPTIONS'
        if l == 'FOOD#STYLE&OPTIONS':
            l = 'FOOD#STYLE_OPTIONS'
        if l in data:
            target.append(1)
        else:
            target.append(0)
    return target


def split(data):
    if isinstance(data, float):
        return np.nan
    return data.split('~')[:-1]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Extract Aspect from Data')
    parser.add_argument('--domain', '--dm', type=str, required=True,
                        help='Tên domain', choices=['restaurant', 'hotel'])

    parser.add_argument('--languague', '--lang', type=str,
                        required=True, help='Ngôn ngữ', choices=['en', 'dutch'])

    parser.add_argument('--type', '--tp', type=str,
                        required=True, help='Train/Dev/Test', choices=['train', 'dev', 'test'])

    args = parser.parse_args()

    with open('semeval/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    domain = args.domain
    lang = args.languague
    typeCSV = args.type + '_csv'
    typeAspect = typeCSV + '_aspects'
    if (domain == 'restaurant'):
        path = config[domain][lang][typeCSV]

        data = pd.read_csv(path, encoding='ISO-8859-1')
        label = ['AMBIENCE#GENERAL', 'DRINKS#PRICES', 'DRINKS#QUALITY', 'DRINKS#STYLE&OPTIONS', 'FOOD#PRICES', 'FOOD#QUALITY',
                 'FOOD#STYLE&OPTIONS', 'LOCATION#GENERAL', 'RESTAURANT#GENERAL', 'RESTAURANT#MISCELLANEOUS', 'RESTAURANT#PRICES', 'SERVICE#GENERAL']

        print('Columns:', data.columns)
        print('Info', data.info())

        cateData = data[['text', 'category']]
        cateData['category'] = cateData['category'].apply(split)
        cateData.dropna(inplace=True)
        cateData = cateData.reset_index(drop=True)

        print('CateData:', cateData.info())
        print('==========================================================')
        dict = set()
        for cate in cateData['category']:
            for item in cate:
                dict.add(item)

        print('Dictionary', dict, '\n', len(dict))

        print('==========================================================')

        cateData['target'] = cateData['category'].apply(
            get_label_restaurant, label=label)

        target = pd.DataFrame(cateData['target'].tolist(), columns=label)
        print('==========================================================')

        print('Target:', target.info())
        print('Target Example', target.head(3))

        cateData = pd.concat([cateData, target], axis=1)
        cateData['Review'] = cateData['text']

        columns = ['Review', 'AMBIENCE#GENERAL', 'DRINKS#PRICES', 'DRINKS#QUALITY',
                   'DRINKS#STYLE&OPTIONS', 'FOOD#PRICES', 'FOOD#QUALITY',
                   'FOOD#STYLE&OPTIONS', 'LOCATION#GENERAL', 'RESTAURANT#GENERAL',
                   'RESTAURANT#MISCELLANEOUS', 'RESTAURANT#PRICES', 'SERVICE#GENERAL',
                   ]
        print('==========================================================')
        print('CateData:', cateData.info())

        print('==========================================================')
        lastData = cateData[columns]
        print('LastData:', lastData.info())

        lastData.to_csv(config[domain][lang]
                        [typeAspect], index=False)
