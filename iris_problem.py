from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


iris_dataset = load_iris()

print("Target names: {}".format(iris_dataset['target_names']))

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'],
                                                    iris_dataset['target'],
                                                    random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

X_new = np.array([[5, 2.9, 1, 0.2]])
prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(iris_dataset['target_names']
                                         [prediction]))

y_pred = knn.predict(X_test)
print("Test set predictions ml:\n {}".format(y_pred))
print("Test set predictions test:\n {}".format(y_test))
print("Test set score:{:.2f}".format(knn.mean(y_pred, y_test)))
