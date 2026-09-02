class MyModel:
    """Plain Python object: no ML framework, no serialized file, just a predict-like method."""

    def predict(self, X):
        # X: a list of rows; each row is a list of two numbers [a, b].
        return [a + b for a, b in X]
