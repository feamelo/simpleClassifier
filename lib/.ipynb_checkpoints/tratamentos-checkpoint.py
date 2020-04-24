from sklearn.base import TransformerMixin
from sklearn.preprocessing import normalize

class Normalizador(TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self):
        pass

    
    def transform(self, X, y=None):
        return normalize(X)


    def fit_transform(self, X, y=None):
        return self.transform(X)