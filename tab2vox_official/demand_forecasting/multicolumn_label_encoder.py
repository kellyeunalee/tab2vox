
import collections
from sklearn.preprocessing import LabelEncoder

class MultiColumnLabelEncoder:

    def __init__(self, columns = None):
        self.columns = columns  
        self.encoders = None

    def fit(self, X, y=None):
        self.encoders = collections.defaultdict(LabelEncoder) 
        if self.columns is not None:                         
            for col in self.columns:
                le = LabelEncoder()
                self.encoders[col] = le.fit(X[col])
        else:
            for colname, col in X.iteritems():
                le = LabelEncoder()
                self.encoders[colname] = le.fit(col)
        return self

    def transform(self, X):     
        output = X.copy()       
        if self.columns is not None:
            for col in self.columns:
                output[col] = self.encoders[col].transform(output[col])
        else:
            for colname, col in output.iteritems():
                output[colname] = self.encoders[col].transform(col)
        return output

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X):
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = self.encoders[col].inverse_transform(output[col])
        else:
            for colname, col in output.iteritems():
                output[colname] = self.encoders[col].inverse_transform(col)
        return output
