"""Train a tiny scikit-learn model and save it as model.pkl.

No external dataset, no network access - the training data is
a handful of hardcoded points, just enough for the model to fit.
"""
import pickle

from sklearn.tree import DecisionTreeClassifier

X = [[0, 0], [1, 1], [2, 2], [3, 3]]
y = [0, 0, 1, 1]

model = DecisionTreeClassifier().fit(X, y)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Saved model.pkl")
