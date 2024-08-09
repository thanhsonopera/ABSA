
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
import json


def aspect_eval(y_test, y_pred, num, save_pred=True, type='val'):
    """
    y_test: grouth_true test
    y_pred: grouth_true predict
    """

    aspect_report = classification_report(
        y_test, y_pred, digits=4, zero_division=1, output_dict=True)

    with open('result/{}/aspect_detection_report_{}.json'.format(type, num), 'w') as f:
        json.dump(aspect_report, f)

    if save_pred:
        with open('checkpoint/aspect_detection_report.json', 'w') as f:
            json.dump(aspect_report, f)


def cus_confusion_matrix(y_test, y_pred, num, save_pred=True, type='val'):

    labels = ['AMBIENCE', 'QUALITY', 'PRICES', 'LOCATION', 'SERVICE']

    cms = multilabel_confusion_matrix(y_test, y_pred)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, (confusion_matrix, ax) in enumerate(zip(cms, axes.flatten())):
        disp = ConfusionMatrixDisplay(confusion_matrix)
        disp.plot(include_values=True, cmap="viridis",
                  ax=ax, xticks_rotation="vertical")
        ax.set_title('Confusion matrix for {}'.format(labels[i]))
        if (i == 4):
            break

    plt.tight_layout()
    # plt.show()
    plt.savefig('result/{}/confusion_matrix_{}.png'.format(type, num))

    if save_pred:
        plt.savefig('checkpoint/confusion_matrix.png')

    plt.close(fig)
