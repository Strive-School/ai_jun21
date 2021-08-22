import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import pandas as pd

class KNN:

    def __init__(self, k):
        self.k = k

    def add_reference(self, x, y):

        self.x = x
        self.y = y

    def ecludian(self, v1, v2):
        assert v1.shape == v2.shape
        return np.sqrt( ((v1-v2) **2).sum())

    def predict(self, x_test):

        self.y_h = []

        for i in range(x_test.shape[0]):
            dif = []
            for j in range(self.x.shape[0]):
                
                euc = self.ecludian(x_test[i], self.x[j])
                dif.append( [ euc, int(self.y[j]) ] )
            
            df = pd.DataFrame(dif, columns=['dist','target'])
            ord_dif = df.sort_values(by='dist')
            self.y_h.append( ord_dif['target'][:self.k].mode()[0] ) 
        return self.y_h

    def evaluate(self, y_test):

        self.acc = 0
        for i in range(y_test.shape[0]):
            if y_test[i] == self.y_h[i]:
                self.acc += 1

        return self.acc / y_test.shape[0]


x, y = make_blobs(n_samples=300, n_features= 2, cluster_std = 0.6, random_state = 0, centers = 4 )
knn = KNN(5)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

knn.add_reference(x_train, y_train)

predictions = knn.predict(x_test)

acc = knn.evaluate(y_test)
print(acc)