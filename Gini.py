# importing libraries and methods

import os
import pandas as pd
import numpy as np
from sklearn.tree._tree import TREE_UNDEFINED
import pydotplus
from sklearn import preprocessing
from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import KFold


os.environ["PATH"] += os.pathsep + "C:/Users/Pooria/Desktop/release/bin/"


balance_data = pd.read_csv(
    os.path.join('data', 'agaricus-lepiota.data'),
    sep=',', header=None)


# Dataset
print("Dataset:")
print(balance_data.head())

#  Encoding
le = preprocessing.LabelEncoder()
balance_data = balance_data.apply(le.fit_transform)

#  KFold
X = balance_data.values[:, 1:23]
Y = balance_data.values[:, 0]
kf = KFold(n_splits=5)
kf.get_n_splits(X)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]



classifier_gini = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=20,
                                              max_features=None, max_leaf_nodes=None, min_samples_leaf=2,
                                              min_samples_split=2, min_weight_fraction_leaf=0.0,
                                              presort=False, random_state=100, splitter='best')
classifier_gini.fit(X_train, y_train)


# Visualizing the tree
dot_data = StringIO()
features_names = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment',
                 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root',
                 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring',
                 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
                 'ring-type', 'spore-print-color', 'population', 'habitat']
tree.export_graphviz(classifier_gini, out_file=dot_data,
                     feature_names=features_names,
                     class_names=['e', 'p'],
                     filled=True, rounded=True,
                     special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("outputs/Gini.pdf")

# prediction with Gini
y_prediction = classifier_gini.predict(X_test)

# Accuracy
print("Accuracy is ", accuracy_score(y_test, y_prediction),'\n')

print(classification_report(y_test, y_prediction))


def rules(tree):
    t = tree.tree_
    rules = []

    def dfs(node, conditions):
        if t.feature[node] != TREE_UNDEFINED:
            fname = features_names[t.feature[node] + 1]
            threshold = t.threshold[node]
            dfs(t.children_left[node], conditions + ["{} <= {}".format(fname, threshold)])
            dfs(t.children_right[node], conditions + ["{} > {}".format(fname, threshold)])
        else:
            class_name = tree.classes_[np.argmax(t.value[node])]
            rules.append("IF {} THEN edible = {}".format(' AND '.join(conditions), class_name))

    dfs(0, [])
    return rules


with open(os.path.join(os.getcwd(),'outputs', 'GiniRULES.txt'), 'w+') as file:
    rules = rules(classifier_gini)
    file.write('\n'.join(rules))










