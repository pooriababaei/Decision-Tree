import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report



ds = pd.read_table(os.path.join( 'data', 'agaricus-lepiota.data'), delimiter=',', header=None)
column_labels = [
    'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruised', 'odor',
    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
    'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
    'stalk-surface-below-ring', 'stalk-color-above-ring',
    'stalk-color below-ring', 'veil-type', 'veil-color', 'ring-number',
    'ring-type', 'spore-print-color', 'population', 'habitat'
]

ds.columns = column_labels
ds = ds[ds['stalk-root'] != '?']
X = ds.loc[:, ds.columns != 'class']
y = ds['class'].to_frame()
y['class'].value_counts()

X_enc = pd.get_dummies(X)

scaler = StandardScaler()
X_std = scaler.fit_transform(X_enc)
le = LabelEncoder()
y_enc = le.fit_transform(y.values.ravel())
X_train, X_test, y_train, y_test = train_test_split(
    X_std,
    y_enc,
    stratify=y_enc,
    test_size=0.2,
    random_state=50
)

print("DataSet:")
print(y_test,'\n')

for i in range (1, 40):
        print('K is ',i,'\n');
        knn = KNeighborsClassifier(n_neighbors = i)
        knn.fit(X_train, y_train)
        y_prediction = knn.predict(X_test)
        print("Classifier Predictions:")
        print(y_prediction,'\n')
        print ('acuuracy is',accuracy_score(y_test, y_prediction),'\n')
        print(classification_report(y_test, y_prediction))
        print('\n\n')
