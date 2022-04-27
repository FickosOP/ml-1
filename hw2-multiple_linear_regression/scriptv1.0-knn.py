import numpy as np
import pandas as pd


def target_encoding(df, target, xi):
    # df = pd.read_csv("data/train.csv", sep=",", index_col=False)
    stats = df[target].groupby(df[xi]).agg(['count', 'mean'])
    smoothing_factor = 1.0
    min_samples_leaf = 1
    prior = df[target].mean()
    smoove = 1 / (1 + np.exp(-(stats['count'] - min_samples_leaf) / smoothing_factor))
    smoothing = prior * (1 - smoove) + stats['mean'] * smoove
    encoded = pd.Series(smoothing)
    return encoded


def load_set(path, is_test):
    df = pd.read_csv(path, sep=",", index_col=False)
    pd.set_option('display.max_columns', None)

    encoded_gender = target_encoding(df, 'plata', 'pol')
    df['pol'].replace(['Female', 'Male'], encoded_gender, inplace=True)

    encoded_field = target_encoding(df, 'plata', 'oblast')
    df['oblast'].replace(['A', 'B'], encoded_field, inplace=True)

    if not is_test:
        df['anomaly'] = detect_outliers(df.iloc[:, :6])
        df = df[df.anomaly != -1]
        df = df.drop(['anomaly'], axis=1)

    encoded_title = target_encoding(df, 'plata', 'zvanje')
    df['zvanje'].replace(['AssocProf', 'AsstProf', 'Prof'], encoded_title, inplace=True)

    x = df.iloc[:, :5].values
    y = df.iloc[:, 5].values

    return np.array(x, dtype=np.float64), np.array(y, dtype=np.float64)


def detect_outliers(df):
    """
    Function for detecting outliers in dataframe
    :param df: Dataframe with all data
    :return: Array of values [-1, 1], -1 meaning observation is outlier 1 meaning it is not
    """
    arr = []
    for index, row in df.iterrows():
        if row['zvanje'] == 'Prof':
            if row['plata'] > 190000 or row['plata'] < 70000:
                arr.append(-1)
            else:
                arr.append(1)
        elif row['zvanje'] == 'AssocProf':
            if row['plata'] > 125000:
                arr.append(-1)
            else:
                arr.append(1)
        else:
            arr.append(1)
    return arr


def calculate_rmse(predicted, true):
    return np.sqrt(((predicted - true) ** 2).mean())


def knn(x_test, y_train, y_test, sorted_distance, k):
    y_pred = np.zeros(y_test.shape)
    for row in range(len(x_test)):
        # Transforming the y_train values to adjust the scale.
        y_pred[row] = y_train[sorted_distance[:, row][:k]].mean()
    return y_pred


def main(train_path, test_path):
    x, y = load_set(train_path, False)

    x_test, y_true = load_set(test_path, True)

    distance = np.sqrt(((x[:, :, None] - x_test[:, :, None].T) ** 2).sum(1))

    # Sorting each data points of the distance matrix to reduce computational effort
    sorted_distance = np.argsort(distance, axis=0)

    y_pred_man_knn = knn(x_test, y, y_true, sorted_distance, 20)

    err = calculate_rmse(y_pred_man_knn, y_true)
    print(err)


if __name__ == '__main__':
    main("data/train.csv", "data/testiramo.csv")
