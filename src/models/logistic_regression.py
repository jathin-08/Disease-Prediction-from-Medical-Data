from sklearn.linear_model import LogisticRegression

class LogisticRegressionModel:
    def __init__(self, C=1.0, max_iter=1000, random_state=42, **kwargs):
        """Initializes L2-regularized Logistic Regression model."""
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            **kwargs
        )
        
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
        
    def predict(self, X):
        return self.model.predict(X)
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)
