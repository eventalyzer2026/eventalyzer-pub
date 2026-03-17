import joblib
import logging
import os
import sys
import numpy as np
import pandas as pd
from scipy import sparse
from pandas.api.types import is_sparse
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class TwoStageLogReg:
    """
    Compatibility class for loading legacy notebooks models serialized as
    __mp_main__.TwoStageLogReg.
    """

    def __init__(self):
        self.anomaly_threshold = 0.40
        self.anomaly_label = -1
        self.anomaly_pipe = None
        self.normal_pipe = None
        self.single_normal_class_ = None
        self.classes_ = None

    @property
    def n_features_in_(self):
        if self.anomaly_pipe is not None:
            return getattr(self.anomaly_pipe, "n_features_in_", None)
        if self.normal_pipe is not None:
            return getattr(self.normal_pipe, "n_features_in_", None)
        return None

    @property
    def feature_names_in_(self):
        if self.anomaly_pipe is not None:
            return getattr(self.anomaly_pipe, "feature_names_in_", None)
        if self.normal_pipe is not None:
            return getattr(self.normal_pipe, "feature_names_in_", None)
        return None

    def anomaly_proba(self, X):
        if self.anomaly_pipe is None:
            return np.zeros(X.shape[0], dtype=float)
        classes = list(getattr(self.anomaly_pipe, "classes_", []))
        if 1 in classes:
            idx = classes.index(1)
            return self.anomaly_pipe.predict_proba(X)[:, idx]
        if len(classes) == 2:
            # fallback for binary encodings that are not {0,1}
            return self.anomaly_pipe.predict_proba(X)[:, -1]
        return np.zeros(X.shape[0], dtype=float)

    def predict(self, X):
        p_anom = self.anomaly_proba(X)
        thr = float(getattr(self, "anomaly_threshold", 0.40))
        anomaly_label = int(getattr(self, "anomaly_label", -1))

        if getattr(self, "normal_pipe", None) is not None:
            normal_pred = self.normal_pipe.predict(X)
        elif getattr(self, "single_normal_class_", None) is not None:
            normal_pred = np.full(X.shape[0], self.single_normal_class_)
        else:
            normal_pred = np.full(X.shape[0], anomaly_label)

        pred = np.asarray(normal_pred).copy()
        pred[p_anom >= thr] = anomaly_label
        return pred

    def predict_proba(self, X):
        classes = getattr(self, "classes_", None)
        if classes is None:
            classes = [int(getattr(self, "anomaly_label", -1))]
            normal_pipe = getattr(self, "normal_pipe", None)
            if normal_pipe is not None:
                classes.extend(list(getattr(normal_pipe, "classes_", [])))
            elif getattr(self, "single_normal_class_", None) is not None:
                classes.append(self.single_normal_class_)
            classes = np.unique(np.asarray(classes))
            self.classes_ = classes
        classes = np.asarray(classes)

        out = np.zeros((X.shape[0], len(classes)), dtype=float)
        class_to_idx = {c: i for i, c in enumerate(classes.tolist())}

        anomaly_label = int(getattr(self, "anomaly_label", -1))
        anomaly_idx = class_to_idx.get(anomaly_label)
        p_anom = self.anomaly_proba(X)
        p_normal = 1.0 - p_anom

        if anomaly_idx is not None:
            out[:, anomaly_idx] = p_anom

        normal_pipe = getattr(self, "normal_pipe", None)
        if normal_pipe is not None:
            normal_classes = list(getattr(normal_pipe, "classes_", []))
            normal_proba = normal_pipe.predict_proba(X)
            for idx, cls in enumerate(normal_classes):
                out_idx = class_to_idx.get(cls)
                if out_idx is not None:
                    out[:, out_idx] = p_normal * normal_proba[:, idx]
        elif getattr(self, "single_normal_class_", None) is not None:
            out_idx = class_to_idx.get(self.single_normal_class_)
            if out_idx is not None:
                out[:, out_idx] = p_normal

        row_sum = out.sum(axis=1, keepdims=True)
        valid = row_sum[:, 0] > 0
        out[valid] = out[valid] / row_sum[valid]
        return out


def _register_legacy_pickle_symbols() -> None:
    for mod_name in ("__mp_main__", "__main__"):
        module = sys.modules.get(mod_name)
        if module is not None and not hasattr(module, "TwoStageLogReg"):
            setattr(module, "TwoStageLogReg", TwoStageLogReg)


_register_legacy_pickle_symbols()


class LogClassificatorModel:
    def __init__(self, model_filepath: str = None):
        if model_filepath is None or model_filepath == "":
            self.pipe = Pipeline([
                ("scaler", StandardScaler(with_mean=False)),
                ("clf", LogisticRegression(
                    max_iter=3000,
                    n_jobs=-1,
                    solver="saga"   # для sparse обычно стабильнее
                ))
            ])
        else:
            self.pipe = joblib.load(model_filepath)

        # Allow runtime tuning for two-stage models restored from artifacts.
        env_threshold = os.getenv("EVENTALYZER_ANOMALY_THRESHOLD")
        if env_threshold is not None and hasattr(self.pipe, "anomaly_threshold"):
            try:
                self.pipe.anomaly_threshold = float(env_threshold)
                logging.info("Applied EVENTALYZER_ANOMALY_THRESHOLD=%s", env_threshold)
            except ValueError:
                logging.warning("Invalid EVENTALYZER_ANOMALY_THRESHOLD=%r", env_threshold)

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

    def predict_anomaly_proba(self, data: pd.DataFrame):
        """Probability for anomaly class (-1) when available."""
        data = self._align_features(data)
        model_input = self._to_model_input(data)

        if hasattr(self.pipe, "anomaly_proba"):
            return np.asarray(self.pipe.anomaly_proba(model_input), dtype=float)

        if hasattr(self.pipe, "predict_proba"):
            proba = self.pipe.predict_proba(model_input)
            classes = getattr(self.pipe, "classes_", None)
            if classes is not None:
                classes = list(classes)
                if -1 in classes:
                    return np.asarray(proba[:, classes.index(-1)], dtype=float)
            return np.zeros(proba.shape[0], dtype=float)

        raise AttributeError("Underlying classifier does not support anomaly probability estimates")

    def dump_model(self, filename: str) -> list:
        return joblib.dump(self.pipe, filename)
