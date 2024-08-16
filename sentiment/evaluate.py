
from sklearn.metrics import classification_report
import pandas as pd
import json
import os


class PolarityMapping:
    INDEX_TO_POLARITY = {0: None, 1: 'positive', 2: 'negative', 3: 'neutral'}
    INDEX_TO_ONEHOT = {0: [1, 0, 0, 0], 1: [
        0, 1, 0, 0], 2: [0, 0, 1, 0], 3: [0, 0, 0, 1]}
    POLARITY_TO_INDEX = {None: 0, 'positive': 1, 'negative': 2, 'neutral': 3}


class Evaluator:
    def __init__(self, y_test, y_pred, aspect_category_names, save_pred=False, num=0, type='val'):
        aspect_cate_test, aspect_cate_pred = [], []
        aspect_cate_polar_test, aspect_cate_polar_pred = [], []

        for row_test, row_pred in zip(y_test, y_pred):
            for index, (col_test, col_pred) in enumerate(zip(row_test, row_pred)):
                aspect_cate_test.append(
                    aspect_category_names[index] if col_test != 0 else 'None#None')
                aspect_cate_pred.append(
                    aspect_category_names[index] if col_pred != 0 else 'None#None')
                aspect_cate_polar_test.append(
                    aspect_category_names[index] + f',{PolarityMapping.INDEX_TO_POLARITY[col_test]}')
                aspect_cate_polar_pred.append(
                    aspect_category_names[index] + f',{PolarityMapping.INDEX_TO_POLARITY[col_pred]}')

        self.aspect_cate_polar_report = classification_report(
            aspect_cate_polar_test, aspect_cate_polar_pred, output_dict=True, zero_division=1)

        self.aspect_cate_report = classification_report(
            aspect_cate_test, aspect_cate_pred, output_dict=True, zero_division=1)

        self.polarity_report = classification_report(y_test.flatten(), y_pred.flatten(
        ), target_names=PolarityMapping.POLARITY_TO_INDEX, output_dict=True)

        print(len(aspect_cate_polar_test), len(aspect_cate_polar_pred))

        path_as_polarity = 'result/{}/aspect_cate_polar_report_{}.json'.format(
            type, num)

        if not os.path.exists(path_as_polarity):
            os.makedirs(os.path.dirname(path_as_polarity), exist_ok=True)

        with open(path_as_polarity, 'w') as f:
            json.dump(self.aspect_cate_polar_report, f)

        if save_pred:
            with open('checkpoint/aspect_cate_polar_report.json', 'w') as f:
                json.dump(self.aspect_cate_polar_report, f)

        self._merge_all_reports()
        self._build_macro_avg_df()

    def report(self, report_type='all'):
        if report_type.lower() == 'all':
            pass
        elif report_type.lower() == 'aspect#category,polarity':
            return pd.DataFrame(self.aspect_cate_polar_report).T
        elif report_type.lower() == 'aspect#category':
            return pd.DataFrame(self.aspect_cate_report).T
        elif report_type.lower() == 'polarity':
            return pd.DataFrame(self.polarity_report).T
        elif report_type.lower() == 'macro_avg':
            return self.macro_avg_df()
        else:
            raise ValueError(
                'report_type must be in ["all", "aspect#category,polarity", "aspect#category", "polarity", "macro_avg"]')

    def _merge_all_reports(self):
        self.merged_report = {}
        for key, metrics in self.aspect_cate_polar_report.items():
            # Check if key in the form of 'aspect#category,polarity' (Check if it's not 'accuracy' or 'macro avg' or 'weighted avg')
            if key in ['accuracy', 'macro avg', 'weighted avg']:
                self.merged_report[key] = {
                    'aspect#category': self.aspect_cate_report[key],
                    'aspect#category,polarity': metrics
                }
            else:
                aspect_cate, polarity = key.split(',')
                if aspect_cate not in self.merged_report:
                    self.merged_report[aspect_cate] = {
                        'aspect#category': self.aspect_cate_report[aspect_cate]}
                self.merged_report[aspect_cate][polarity] = metrics

    def _build_macro_avg_df(self):
        self.macro_avg_df = pd.DataFrame([{
            'accuracy': f"{report['accuracy']:.3f}",
            # **{metric: report['macro avg'][metric] for metric in report['macro avg'] if metric != 'accuracy'}
            'precision': f"{report['macro avg']['precision']:.3f}",
            'recall': f"{report['macro avg']['recall']:.3f}",
            'f1-score': f"{report['macro avg']['f1-score']:.3f}",
            'support': report['macro avg']['support']
        } for report in [self.aspect_cate_polar_report, self.aspect_cate_report, self.polarity_report]])
        self.macro_avg_df.index = [
            'Aspect#Category,Polarity', 'Aspect#Category', 'Polarity']
        print(self.macro_avg_df)
