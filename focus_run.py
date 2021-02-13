from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
message = data.data
related = data.target

# splitting for train and test
x_train, x_test, y_train, y_test = train_test_split(message, related, train_size=0.9)

# Using lazy predict
clf = LazyClassifier(classifiers='all')
model, predictions = clf.fit(x_train, x_test, y_train, y_test)
print(model)
