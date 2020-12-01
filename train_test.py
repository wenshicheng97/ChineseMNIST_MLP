import numpy as np
import pandas as pd

labels_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 100: 11, 1000: 12, 10000: 13,
               100000000: 14}
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1000, 10000, 100000000]


def train_neuralnetwork(n, rounds=2):
    np.random.seed(0)
    train_df = pd.read_csv("train_set.csv", header=None)
    train_df = train_df.reindex(np.random.permutation(train_df.index)).reset_index(drop=True)

    for e in range(rounds):
        print('round:', e+1)
        for index, row in train_df.iterrows():
            feature = row.drop([4096]).values
            inputs = (np.asfarray(feature) / 255.0 * 0.99) + 0.01
            targets = np.zeros(15) + 0.01
            targets[labels_dict[row[4096]]] = 0.99
            n.train(inputs, targets)

    pass


def test_neuralnetwork(n):
    test_df = pd.read_csv("test_set.csv", header=None)
    counter = 0
    for index, row in test_df.iterrows():
        feature = row.drop([4096]).values
        inputs = (np.asfarray(feature) / 255.0 * 0.99) + 0.01
        true_label = labels_dict[row[4096]]
        outputs = n.check(inputs)
        counter += (true_label == np.argmax(outputs))

    print('accuracy:', counter/test_df.shape[0])


def predict_neuralnetwork(n):
    test_df = pd.read_csv("test_set.csv", header=None)
    real = []
    pred = []
    for index, row in test_df.iterrows():
        feature = row.drop([4096]).values
        inputs = (np.asfarray(feature) / 255.0 * 0.99) + 0.01
        true_label = labels_dict[row[4096]]
        outputs = n.check(inputs)
        real.append(true_label)
        pred.append(np.argmax(outputs))

    return np.array(real), np.array(pred)
