import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

import util

CSV_PATH = "../../full_data_one_hot.csv"


def svm_regression(inputs, labels):
    print(inputs.shape)
    # Fit the model
    labels_train = labels[:400000]
    labels_test = labels[400000:500000]
    inputs_train = inputs[:400000]
    inputs_test = inputs[400000:500000]

    print("starting to fit")

    regr = svm.SVR(kernel='linear')
    regr.fit(inputs_train, labels_train)
    print('successfully fit!')

    y_pred = regr.predict(inputs_test)
    y_pred_trained = regr.predict(inputs_train)
    print("got y_pred")
    # acc = regr.score(inputs_test, labels_test)
    # print(f"accuracy: {acc}")

    # find absolute value median score
    abs_score = np.sum(np.absolute(y_pred - labels_test))/len(inputs_test)
    abs_score_train = np.sum(np.absolute(
        y_pred_trained - labels_train))/len(inputs_train)
    print(f"median absolute deviance: {abs_score}")
    print(f"median absolute deviance trained: {abs_score_train}")

    # plot accuracy
    plt.scatter(labels_test, y_pred)
    plt.xlabel('y_true')
    plt.ylabel('y_predictions')
    plt.title('SVM plot')
    plt.savefig('svm_plot3.png')


def main():
    inputs, labels = util.load_dataset(CSV_PATH)
    svm_regression(inputs, labels)


if __name__ == '__main__':
    main()
