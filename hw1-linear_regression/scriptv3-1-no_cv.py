import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_train_set(path):
    """
    :param path: location of data set
    :return: array of X values, array of transformed Y values
    """
    df = pd.read_csv(path, sep=",", index_col=False)
    df.columns = ["x", "y"]

    # remove outliers
    for index, row in df.iterrows():
        if row['x'] < 0.4 and row['y'] > 35:
            df.drop([index], inplace=True)

    df.insert(len(df.columns), 'BoxCox', bc(df.iloc[:, 1]))

    data = np.array(df, dtype=float)

    return np.array(data[:, 0]), np.array(data[:, -1])


def load_test_set(path):
    """
    :param path: location of data set
    :return: array of X values, array of Y values
    """
    df = pd.read_csv(path, sep=",", index_col=False)
    df.columns = ["x", "y"]

    data = np.array(df, dtype=float)

    return np.array(data[:, 0]), np.array(data[:, -1])


def bc(arr, ld=0.07755305496222023):
    """
    Box cox transformation function
    :param arr: Array of values to transform
    :param ld: Lambda that is used for calculation.
    :return: Transformed array
    """
    if ld == 0:
        return np.log(arr)
    return (arr ** ld - 1)/ld


def i_bc(y, ld=0.07755305496222023):
    """
    :param y: Array of values to transform
                otherwise transformation will produce incorrect values.
    :return: Transformed array
    """
    if ld == 0:
        return np.exp(y)
    else:
        return np.exp(np.log(ld * y + 1) / ld)


def cost_function(m, c, x, y):
    """
    y = m*x + c
    :param m: Slope
    :param c: Intercept
    :param x: Array of X values
    :param y: Array of Y values
    :return: RSS value
    """
    return sum((h(x, m, c) - y) ** 2)/(len(x))


def h(x, m, c):
    return m*x + c


def plot_error_function(j, i):
    plt.scatter(j, i)
    plt.show()


def gradient_descent(x, y, learning_rate=0.1, max_iterations=1000, stopping_threshold=1e-9):
    j_all = []
    iterations = []
    m = 1
    c = 0
    previous_cost = None
    for _ in range(max_iterations):
        current_cost = cost_function(m, c, x, y)
        j_all.append(current_cost)
        iterations.append(m)
        if previous_cost and previous_cost - current_cost < stopping_threshold:
            break
        previous_cost = current_cost

        h_x = h(x, m, c)
        m = m - (learning_rate * sum((h_x - y) * x)/len(x))
        c = c - (learning_rate * sum((h_x - y))/len(x))

    plot_error_function(j_all, iterations)
    return m, c


def fit(x, y):
    m, c = gradient_descent(x, y, learning_rate=0.1, max_iterations=10000)
    return m, c


def predict(x, m, c):
    """
    :param x: X values that prediction is based on
    :param m: Slope
    :param c: Intercept
    :return: Predicted y values
    """
    return i_bc(h(x, m, c))


def calculate_rmse(predicted, true):
    """
    Formula to calculate Residual Mean Squared Error
    :param predicted: Values received from predict function
    :param true: Actual values
    :return: RMSE
    """
    return np.sqrt(((predicted - true) ** 2).mean())


def plot_line(m, c, x, y):
    l_space = np.linspace(0, 0.5, 100)
    func = h(l_space, m, c)

    plt.plot(l_space, func, '-r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(x, y)
    plt.show()


def main(train_path, test_path):
    x, y = load_train_set(train_path)

    m, c = fit(x, y)

    plot_line(m, c, x, y)

    x_test, y_true = load_test_set(test_path)

    y_predicted = predict(x_test, m, c)

    # plot_line(m, c, x_test, y_predicted)

    error = calculate_rmse(y_predicted, y_true)

    print(error)


if __name__ == '__main__':
    main("data/train.csv", "data/test_preview.csv")
