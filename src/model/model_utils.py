from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator

class ModelWrapper(ABC):
    """Abstract base class for model wrappers"""
    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass

    def save(self, path):
        """Optional: save model to path"""
        pass

    def load(self, path):
        """Optional: load model from path"""
        pass

class SklearnWrapper(ModelWrapper):
    def __init__(self, model_class, **params):
        super().__init__(**params)
        self.model = model_class(**params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, path):
        import joblib
        joblib.dump(self.model, path)

    def load(self, path):
        import joblib
        self.model = joblib.load(path)