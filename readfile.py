import cv2
import pandas as pd
import numpy as np


def convert_images():
    df = pd.read_csv('chinese_mnist.csv')
    counter = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 100: 0, 1000: 0, 10000: 0,
               100000000: 0}
    data = []
    train = []
    test = []
    for i in range(df.shape[0]):
        image = cv2.imread('data/input_%d_%d_%d.jpg' % (df.loc[i, 'suite_id'], df.loc[i, 'sample_id'],
                                                      df.loc[i, 'code']), cv2.IMREAD_GRAYSCALE)
        flat_image = image.flatten()
        row = np.append(flat_image, df.loc[i, 'value'])
        data.append(row)
        if counter[df.loc[i, 'value']] < 800:
            train.append(row)
        else:
            test.append(row)
        counter[df.loc[i, 'value']] += 1
    data = pd.DataFrame(data=data)
    train = pd.DataFrame(data=train)
    test = pd.DataFrame(data=test)
    data.to_csv('grayscale.csv', header=False, index=False)
    train.to_csv('train_set.csv', header=False, index=False)
    test.to_csv('test_set.csv', header=False, index=False)


def convert_images_reduced():
    df = pd.read_csv('chinese_mnist.csv')
    counter = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 100: 0, 1000: 0, 10000: 0,
               100000000: 0}
    data = []
    train = []
    test = []
    for i in range(df.shape[0]):
        if counter[df.loc[i, 'value']] >= 100:
            continue
        image = cv2.imread('data/input_%d_%d_%d.jpg' % (df.loc[i, 'suite_id'], df.loc[i, 'sample_id'],
                                                      df.loc[i, 'code']), cv2.IMREAD_GRAYSCALE)
        flat_image = image.flatten()
        row = np.append(flat_image, df.loc[i, 'value'])
        data.append(row)
        if counter[df.loc[i, 'value']] < 80:
            train.append(row)
        else:
            test.append(row)
        counter[df.loc[i, 'value']] += 1
    data = pd.DataFrame(data=data)
    train = pd.DataFrame(data=train)
    test = pd.DataFrame(data=test)
    data.to_csv('grayscale_reduced.csv', header=False, index=False)
    train.to_csv('train_set_reduced.csv', header=False, index=False)
    test.to_csv('test_set_reduced.csv', header=False, index=False)