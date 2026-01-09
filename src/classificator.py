import joblib
import pandas as pd
from scipy import sparse
from pandas.api.types import is_sparse
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class LogClassificatorModel:
    def __init__(self, model_filepath: str = None):
        if model_filepath is None or model_filepath == "":
            self.pipe = Pipeline([
                ('scaler', StandardScaler(with_mean=False)),  # IMPORTANT for sparse data
                ('clf', LogisticRegression(
                    max_iter=3000,
                    n_jobs=-1,
                    solver='lbfgs'
                ))
            ])
        else:
            self.pipe = joblib.load(model_filepath)

    def learn(self, data: pd.DataFrame):
        X = data.drop('cluster', axis=1)
        y = data['cluster']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.pipe.fit(X_train, y_train)

    @staticmethod
    def _to_model_input(data: pd.DataFrame):
        # Avoid pandas sparse -> dense warnings by converting to scipy sparse when possible
        if isinstance(data, pd.DataFrame):
            try:
                if any(is_sparse(dtype) for dtype in data.dtypes):
                    coo = data.sparse.to_coo()
                    return sparse.csr_matrix(coo)
            except Exception:
                pass
            try:
                return data.to_numpy()
            except Exception:
                pass
        return data

    def classify(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.pipe.predict(self._to_model_input(data))

    def predict_proba(self, data: pd.DataFrame):
        """Probability estimates for each class."""
        if not hasattr(self.pipe, "predict_proba"):
            raise AttributeError("Underlying classifier does not support probability estimates")
        return self.pipe.predict_proba(self._to_model_input(data))

    def dump_model(self, filename: str) -> list:
        return joblib.dump(self.pipe, filename)
