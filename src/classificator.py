import joblib
import logging
import os
import numpy as np
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

    def _align_features(self, data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            expected_count = getattr(self.pipe, "n_features_in_", None)
            actual_count = None
            try:
                actual_count = data.shape[1]
            except Exception:
                actual_count = None
            if expected_count is not None and actual_count is not None and expected_count != actual_count:
                logging.warning(
                    "Feature count mismatch for non-DataFrame input: expected=%d, got=%d",
                    expected_count,
                    actual_count,
                )
                if actual_count > expected_count:
                    if sparse.issparse(data):
                        data = data[:, :expected_count]
                    else:
                        data = np.asarray(data)[:, :expected_count]
                else:
                    pad = expected_count - actual_count
                    if sparse.issparse(data):
                        data = sparse.hstack(
                            [data, sparse.csr_matrix((data.shape[0], pad))],
                            format="csr",
                        )
                    else:
                        data = np.pad(
                            np.asarray(data),
                            ((0, 0), (0, pad)),
                            mode="constant",
                        )
            return data

        debug_features = os.getenv("EVENTALYZER_DEBUG_FEATURES") == "1"
        expected = None
        expected_count = getattr(self.pipe, "n_features_in_", None)
        scaler = None
        if hasattr(self.pipe, "feature_names_in_"):
            expected = list(self.pipe.feature_names_in_)
            expected_count = len(expected)
        elif hasattr(self.pipe, "named_steps"):
            scaler = self.pipe.named_steps.get("scaler")
            if scaler is not None and hasattr(scaler, "feature_names_in_"):
                expected = list(scaler.feature_names_in_)
                expected_count = len(expected)
            elif scaler is not None and hasattr(scaler, "n_features_in_"):
                expected_count = scaler.n_features_in_

        if not expected:
            if expected_count is not None and expected_count != data.shape[1]:
                logging.warning(
                    "Feature count mismatch without names: expected=%d, got=%d",
                    expected_count,
                    data.shape[1],
                )
                if debug_features:
                    logging.warning("Feature columns sample: %s", list(data.columns)[:40])
            return data

        missing = [col for col in expected if col not in data.columns]
        extra = [col for col in data.columns if col not in expected]
        if missing:
            for col in missing:
                data[col] = 0
        if extra:
            data = data.drop(columns=extra)
        if missing or extra:
            logging.warning(
                "Feature mismatch: expected=%d, got=%d, missing=%d, extra=%d",
                len(expected),
                len(data.columns),
                len(missing),
                len(extra),
            )
            if debug_features:
                logging.warning("Missing sample: %s", missing[:40])
                logging.warning("Extra sample: %s", extra[:40])
        return data[expected]

    def classify(self, data: pd.DataFrame) -> pd.DataFrame:
        data = self._align_features(data)
        return self.pipe.predict(self._to_model_input(data))

    def predict_proba(self, data: pd.DataFrame):
        """Probability estimates for each class."""
        if not hasattr(self.pipe, "predict_proba"):
            raise AttributeError("Underlying classifier does not support probability estimates")
        data = self._align_features(data)
        return self.pipe.predict_proba(self._to_model_input(data))

    def dump_model(self, filename: str) -> list:
        return joblib.dump(self.pipe, filename)
