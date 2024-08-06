
from sklearn.metrics import classification_report
import json


def aspect_eval(y_test, y_pred, num, save_pred):
    """
    y_test: grouth_true test
    y_pred: grouth_true predict
    """

    aspect_report = classification_report(
        y_test, y_pred, digits=4, zero_division=1, output_dict=True)

    with open('result/aspect_detection_report_{}.json'.format(num), 'w') as f:
        json.dump(aspect_report, f)

    if save_pred:
        with open('checkpoint/aspect_detection_report.json', 'w') as f:
            json.dump(aspect_report, f)
