from xgboost import XGBClassifier

class XGBoostModel:
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.3, random_state=42, **kwargs):
        """Initializes XGBoost Classifier with default logloss metrics."""
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            eval_metric="logloss",
            use_label_encoder=False,
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
        
    @property
    def feature_importances_(self):
        return self.model.feature_importances_
