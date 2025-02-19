import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from DecisionTreeClass import DecisionTree
from DecisionTreeNode import DecisionTreeNode
from DecisionTreePlot import DecisionTreeGraphviz

iris = load_iris()

df = pd.DataFrame(iris.data)
df.head()

random_seed = 1
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=random_seed)

clf_self = DecisionTree(criterion = "gini", min_samples_leaf = 10, random_state = random_seed)
clf_self.fit(X_train, y_train)

print(clf_self.tree)

visualizer = DecisionTreeGraphviz(clf_self.tree)
visualizer.visualize("decision_tree")