import numpy as np
import pandas as pd
from unicodedata import category


def read_data() -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv('winequality-red.csv').sample(frac=1, random_state=0)
    x = df.loc[:, 'fixed acidity':'alcohol'].to_numpy()
    y = df['quality'].to_numpy()
    return x, y


# ここに回答を記述
def split_data(x, y):
    n = len(x)
    n1 = int(n * 0.8)
    x1 = x[:n1]
    y1 = y[:n1]
    x2 = x[n1:]
    y2 = y[n1:]
    return x1, y1, x2, y2


def calc_rss(y):
    rss = np.sum((y - np.mean(y)) ** 2)
    return rss


def get_best_gain(x, y, rss):
    best_gain = 0
    best_value = None
    best_column = 0
    for i in range(x.shape[1]):
        th = np.unique(x[:, i])
        for j in th:
            g_a, g_b = y[x[:, i] >= j], y[x[:, i] < j]

            if len(g_a) == 0 or len(g_b) == 0:
                continue
            gain = rss - (calc_rss(g_a) + calc_rss(g_b))

            if gain > best_gain:
                best_gain = gain
                best_value = j
                best_column = i
    return best_gain, best_value, best_column


def calc_r2(y1, yp1):
    r2 = 1 - np.mean(np.square(y1 - yp1)) / np.var(y1)
    return r2


class Node:
    def __init__(self, depth=0):
        self.large_node = None
        self.small_node = None
        self.category = None
        self.threshold = None
        self.label = None
        self.depth = depth

    def train(self, x, y):
        rss = calc_rss(y)
        if rss == 0 or self.depth == 5:
            self.label = y.mean()
            return

        gain, value, column = get_best_gain(x, y, rss)

        if column is None:
            self.label = y.mean()
            return

        self.category = column
        self.threshold = value

        self.large_node = Node(self.depth + 1)
        self.large_node.train(x[x[:, self.category] >= self.threshold], y[x[:, self.category] >= self.threshold])

        self.small_node = Node(self.depth + 1)
        self.small_node.train(x[x[:, self.category] < self.threshold], y[x[:, self.category] < self.threshold])

    def predict(self, datum):
        if self.label is None:
            if datum[self.category] >= self.threshold:
                return self.large_node.predict(datum)
            else:
                return self.small_node.predict(datum)
        else:
            return self.label


class Tree:
    def __init__(self):
        self.node = None

    def train(self, x, y):
        self.node = Node()
        self.node.train(x, y)

    def predict(self, x):
        yp = [self.node.predict(datum) for datum in x]
        return pd.Series(yp)


class GradientBoostingDecisionTree:
    def __init__(self, tree_num):
        self.trees = []
        self.categories = []
        self.learning_rate = 0.1
        self.tree_num = tree_num
        self.initial_prediction = 0

    def train(self, x, y):
        self.initial_prediction = np.mean(y)
        current = np.full_like(y, self.initial_prediction, dtype=np.float64)

        for i in range(self.tree_num):
            residual = y - current

            tree = Tree()
            tree.train(x, residual)
            self.trees.append(tree)

            tp = tree.predict(x)
            current += self.learning_rate * tp.to_numpy()

    def predict(self, x):
        yp = np.full(len(x), self.initial_prediction)
        for tree in self.trees:
            tp = tree.predict(x).to_numpy()
            yp += self.learning_rate * tp
        return pd.Series(yp)


def main():
    np.random.seed(0)
    x, y = read_data()
    x1, y1, x2, y2 = split_data(x, y)
    gbdt = GradientBoostingDecisionTree(80)
    gbdt.train(x1, y1)
    yp1 = gbdt.predict(x1)
    r2_1 = calc_r2(y1, yp1)
    print('Train data:', r2_1)
    yp2 = gbdt.predict(x2)
    r2_2 = calc_r2(y2, yp2)
    print('Test data:', r2_2)


if __name__ == '__main__':
    main()
