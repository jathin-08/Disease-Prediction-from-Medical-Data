from sklearn.svm import SVC

class SVMModel:
    def __init__(self, C=1.0, kernel="rbf", gamma="scale", random_state=42, **kwargs):
        """Initializes SVM model with probability estimates enabled."""
        self.model = SVC(
            C=C, 
            kernel=kernel, 
            gamma=gamma, 
            probability=True, 
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
