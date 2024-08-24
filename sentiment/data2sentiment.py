import numpy as np
import pandas as pd
import argparse
import os
import yaml
# python sentiment/data2sentiment.py --dm hotel --lang vn --tp train --mer default [--dr t]
# python sentiment/data2sentiment.py --dm restaurant --lang vn --tp train --mer default [--dr t]
# python sentiment/data2sentiment.py --dm all --lang vn --tp train --mer default [--dr t]
# python sentiment/data2sentiment.py --dm all --lang vn --tp train --mer traindev [--dr t]


def merge2Setiment(row, category, newCategory, domain='restaurant'):
    init = np.zeros(len(newCategory))
    oldCategory = np.array(category)
    value = np.array(row)[1:]
    dt = pd.DataFrame([init], columns=newCategory, dtype=int)
    if (domain == 'restaurant'):
        for i, col in enumerate(oldCategory):
            for k in newCategory:
                if k in col:
                    if (value[i] > 0):
                        dt.loc[0, k] = value[i] if dt.loc[0, k] != 2 else 2

    elif (domain == 'hotel'):
        for i, col in enumerate(oldCategory):
            for k in newCategory:
                kp = 'XXXXX'
                if k == 'AMBIENCE':
                    kp = 'ROOM_AMENITIES'
                if k == 'QUALITY':
                    kp = 'FACILITIES'
                if (k in col) or (kp in col):
                    if (value[i] > 0):
                        dt.loc[0, k] = value[i] if dt.loc[0, k] != 2 else 2

    return dt.iloc[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Chuyển đổi dữ liệu sentiment')

    parser.add_argument('--domain', '--dm', type=str, required=True,
                        help='Tên domain', choices=['restaurant', 'hotel', 'all'])
    parser.add_argument('--languague', '--lang', type=str,
                        required=True, help='Ngôn ngữ', choices=['vn'])
    parser.add_argument('--type', '--tp', type=str,
                        required=True, help='Train/Dev/Test', choices=['train', 'dev', 'test'])
    parser.add_argument('--drop', '--dr', type=bool, default=False,
                        help='Xóa các dòng không có aspect')
    parser.add_argument('--merge', '--mer', type=str,
                        required=True, help='Hợp file', choices=['traindev', 'traintest', 'default'])

    args = parser.parse_args()
    with open('sentiment/relabel/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    newCategory = ['Review', 'AMBIENCE',
                   'QUALITY', 'PRICES', 'LOCATION', 'SERVICE']

    domain = args.domain
    lang = args.languague
    type = args.type

    if (args.merge != 'default' and domain != 'all'):
        raise Exception('Please choose domain all')

    if (args.merge == 'default' and domain != 'all'):
        path_input = config[domain][lang][type]
        path_output = config[domain][lang][type + '_sentiment']

        if not os.path.exists(os.path.dirname(path_output)):
            os.makedirs(os.path.dirname(path_output), exist_ok=True)

        data = pd.read_csv(path_input)

        newCol = data.apply(lambda row: merge2Setiment(
            row, data.columns[1:], newCategory[1:], domain), axis=1)

        newCol['Review'] = data['Review']
        resultPd = newCol[newCategory]

        if args.drop == True:
            resultPd = resultPd[(resultPd[newCategory[1:]] != 0).any(axis=1)]

        resultPd.to_csv(path_output, index=False)
        print(resultPd.info())

    if (args.merge == 'default' and domain == 'all'):
        path_dm1 = config['restaurant'][lang][type + '_sentiment']
        path_dm2 = config['hotel'][lang][type + '_sentiment']
        path_output = config['all'][lang][type + '_sentiment']
        if os.path.exists(os.path.dirname(path_dm1)) and os.path.exists(os.path.dirname(path_dm2)):
            if not os.path.exists(os.path.dirname(path_output)):
                os.makedirs(os.path.dirname(path_output), exist_ok=True)
            data1 = pd.read_csv(path_dm1)
            data2 = pd.read_csv(path_dm2)
            resultPd = pd.concat([data1, data2], ignore_index=True, axis=0)

            if args.drop == True:
                resultPd = resultPd[(
                    resultPd[newCategory[1:]] != 0).any(axis=1)]

            resultPd.to_csv(path_output, index=False)
            print(resultPd.info())
        else:
            raise Exception(
                'File not found ^ ^. Please domain restaurant or hotel first')

    if (args.merge != 'default' and domain == 'all'):
        if args.merge == 'traindev':
            type1 = 'train'
            type2 = 'dev'
        else:
            type1 = 'train'
            type2 = 'test'
        path_dm1 = config['all'][lang][type1 + '_sentiment']
        path_dm2 = config['all'][lang][type2 + '_sentiment']
        path_output = config['all'][lang][args.merge + '_sentiment']
        if os.path.exists(os.path.dirname(path_dm1)) and os.path.exists(os.path.dirname(path_dm2)):
            if not os.path.exists(os.path.dirname(path_output)):
                os.makedirs(os.path.dirname(path_output), exist_ok=True)
            data1 = pd.read_csv(path_dm1)
            data2 = pd.read_csv(path_dm2)
            resultPd = pd.concat([data1, data2], ignore_index=True, axis=0)

            if args.drop == True:
                resultPd = resultPd[(
                    resultPd[newCategory[1:]] != 0).any(axis=1)]

            resultPd.to_csv(path_output, index=False)
            print(resultPd.info())

        else:
            raise Exception('File not found ^ ^. Please domain all first')
