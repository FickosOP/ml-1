from scipy.stats import zscore
from numpy.linalg import inv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys


def load_train_set(path):
    df = pd.read_csv(path, sep=",", index_col=False)

    df['zvanje'].replace(['Prof', 'AsstProf', 'AssocProf'], [3, 1, 2], inplace=True)
    df['oblast'].replace(['A', 'B'], [1, 2], inplace=True)
    df['pol'].replace(['Female', 'Male'], [0, 1], inplace=True)
    df.insert(len(df.columns), 'y_sqrt', np.sqrt(df.iloc[:, -1]))

    df['zvanje_z'] = zscore(df['zvanje'])
    df['oblast_x'] = zscore(df['oblast'])
    # df['pol_z'] = df['pol']
    df['godina_doktor_z'] = zscore(df['godina_doktor'])
    df['godina_iskustva_z'] = zscore(df['godina_iskustva'])
    # print(df)

    x = df.iloc[:, 7:].values
    y = df.iloc[:, 6].values

    df.hist(grid=False, figsize=(10, 6), bins=30)
    # plt.show()

    # print(df.agg(['skew', 'kurtosis']).transpose())
    # print(df)
    return np.array(x, dtype=np.float64), np.array(y, dtype=np.float64)


def load_test_set(path):
    return 0


def cost_function(theta, x, y):
    return 0


def h(x, theta, b):
    return x.dot(theta) + b


def ridge_regression(x, y, learning_rate=0.1, max_iterations=1000, stopping_threshold=1e-9, l2_penalty=0.5):
    n = len(y)

    diag_matrix = []
    for i in range(x.shape[1]):
        diag_row = []
        for j in range(x.shape[1]):
            if i > 0 and i == j:
                diag_row.append(1)
            else:
                diag_row.append(0)
        diag_matrix.append(diag_row)
    diag_matrix = np.array(diag_matrix, dtype=np.float64)
    # Closed form
    theta = inv((x.T.dot(x) + l2_penalty * diag_matrix)).dot(x.T.dot(y))
    print("THETA")
    print(theta)
    return theta
    # for _ in range(max_iterations):
    #     # print(_)
    #     predicted = np.dot(x, theta)
    #     theta = theta - (1/n)*learning_rate*(x.T.dot(predicted - y)) + l2_penalty/2 * theta.T.dot(theta)
    #
    #     if np.isnan(theta[0]):
    #         print(_)
    #         break
    #
    # return theta


def fit(x, y):
    theta = ridge_regression(x, y)
    return theta


def predict(x, theta):
    # print("PREDICTED\n\n")
    # print(np.power(np.dot(x, theta), 2))
    return np.power(np.dot(x, theta), 2)  # np.dot(x, theta)


def calculate_rmse(predicted, true):
    return np.sqrt(((predicted - true) ** 2).mean())


def create_new_sets(x_parts, y_parts, i):
    new_x, new_y, new_x_val, new_y_true = [], [], [], []
    print(f"LenX = {len(x_parts)}")
    print(f"LenY = {len(y_parts)}")
    # print(y_parts)
    for j in range(len(x_parts)):
        if j == i:
            for row in x_parts[j]:
                new_x_val.append(row.tolist())
            for row in y_parts[j]:
                new_y_true.append(row.tolist())

        else:
            for row in x_parts[j]:
                new_x.append(row.tolist())
            for row in y_parts[j]:
                new_y.append(row.tolist())

    # print(new_x)
    return np.array(new_x, dtype=np.float64),\
           np.array(new_y, dtype=np.float64),\
           np.array(new_x_val, dtype=np.float64),\
           np.array(np.power(new_y_true, 2), dtype=np.float64)  # take it from df


def cross_val(x_b, y, k):
    x_parts = np.array_split(x_b, k)
    y_parts = np.array_split(y, k)

    best_theta = []
    err_list = []
    lowest_err = None
    for i in range(k):
        new_x, new_y, new_x_val, new_y_true = create_new_sets(x_parts, y_parts, i)
        theta = ridge_regression(new_x, new_y)
        y_predicted = predict(new_x_val, theta)
        print("\nPredicted - True\n")
        for _ in range(len(y_predicted)):
            print(f"P: {y_predicted[_]} -- T: {new_y_true[_]}\t Diff: {y_predicted[_] - new_y_true[_]}")
        # print("\n\nTRUE\n\n")
        # print(new_y_true)
        err = calculate_rmse(y_predicted, new_y_true)
        if not lowest_err or err < lowest_err:
            lowest_err = err
            best_theta = theta
        # if len(err_list) == 0 or err < err_list[-1]:
        #     best_theta = theta

        err_list.append(err)
        # wait = input()
    print(f"Error list: {err_list}")
    print(f"Avg error: {sum(err_list)/len(err_list)}")
    return best_theta


def main(train_path, test_path):
    x, y = load_train_set(train_path)

    x_b = np.c_[np.ones((x.shape[0], 1)), x]

    best_theta = cross_val(x_b, y, 10)
    print(f"BEST THETA: {best_theta}")
    #
    # x_test, y_true = load_train_set(test_path)

    # y_predicted = predict(x, theta)
    # error = calculate_rmse(y_predicted, y)
    #
    # print(f"RMSE:{error}")


if __name__ == '__main__':
    main("data/train.csv", "data/test_preview.csv")
    # main(sys.argv[1], sys.argv[2])


