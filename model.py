import numpy as np
from scipy.io import arff
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from datetime import datetime
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.io import arff

from sklearn.metrics import confusion_matrix


def load_arff(file_path):
    data, meta = arff.loadarff(file_path)
    x = []
    y = []
    for w in range(len(data)):
        x.append([])
        for k in range(len(data[0])):
            if k == (len(data[0]) - 1):
                y.append(data[w][k])
            else:
                x[w].append(data[w][k])
    classes = list(set(y))
    return x, y, classes


def classifiers_characteristics(classifier, probes, X, Y):
    accuracy = []
    sensitivity = []
    specificity = []
    precision = []
    F1 = []
    balanced_accuracy = []
    auc = []
    fit_time = []
    predict_time = []

    for i in range(probes):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=i)

        startTime = datetime.now()
        classifier.fit(X_train, y_train)
        timeElapsed = datetime.now() - startTime
        fit_time.append(timeElapsed)

        startTime = datetime.now()
        y_pred = classifier.predict(X_test)
        timeElapsed = datetime.now() - startTime
        predict_time.append(timeElapsed)

        y_decision = classifier.predict_proba(X_test)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        accuracy.append((tp + tn) / (tp + tn + fp + fn))
        sensitivity.append(tp / (tp + fn))
        specificity.append(tn / (fp + tn))
        precision.append(tp / (tp + fp))
        F1.append(2 * (precision[i] * sensitivity[i]) / (precision[i] + sensitivity[i]))
        balanced_accuracy.append(0.5 * (sensitivity[i] + specificity[i]))
        auc.append(roc_auc_score(y_test, y_decision[:, 1]))

    return np.mean(accuracy), np.mean(sensitivity), np.mean(specificity), np.mean(precision), np.mean(F1), np.mean(
        balanced_accuracy), np.mean(auc), np.mean(fit_time), np.mean(predict_time)


def roc_plots(classifiers, names, X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=0)

    for index, classifier in enumerate(classifiers):

        classifier.fit(X_train, y_train)

        y_decision = classifier.predict_proba(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_decision[:, 1])

        cut_off = [1, 0]
        for i in range(len(fpr)):
            cut_off_dist = np.sqrt(np.power(cut_off[0], 2) + np.power((1 - cut_off[1]), 2))
            test = np.sqrt(np.power(fpr[i], 2) + np.power((1 - tpr[i]), 2))
            if test < cut_off_dist:
                cut_off[0] = fpr[i]
                cut_off[1] = tpr[i]

        plt.plot(fpr, tpr, label=names[index])
        plt.plot(cut_off[0], cut_off[1], 'ro', markersize=6)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Krzywe ROC dla badanych klasyfikatorów')
    plt.legend(loc="lower right")
    plt.show()
