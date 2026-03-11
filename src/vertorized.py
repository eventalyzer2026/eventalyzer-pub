import ipaddress
import json
import logging
from functools import lru_cache

import joblib
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


@lru_cache(maxsize=100000)
def _ip_to_int(value: str) -> int:
    try:
        return int(ipaddress.ip_address(value))
    except Exception:
        return 0


class LogVertorizer:
    def __init__(self, ohe_encoding_features: list[str] = None, le_encoding_features: list[str] = None,
                 insufficent_columns: list[str] = None, ohe_filepath: str = None, le_filepath: str = None):
        self.dir_vec = HashingVectorizer(
            analyzer='char',
            ngram_range=(3, 5),
            n_features=2 ** 14,
            alternate_sign=False
        )

        self.file_vec = HashingVectorizer(
            analyzer='char',
            ngram_range=(3, 5),
            n_features=2 ** 12,
            alternate_sign=False
        )

        self.ref_vec = HashingVectorizer(
            analyzer='char',
            ngram_range=(3, 5),
            n_features=2 ** 14,
            alternate_sign=False
        )

        if ohe_encoding_features is None:
            self.ohe_encoding_features = ['http.request.method', 'http.response.status_code', 'user_agent.os.name']
        else:
            self.ohe_encoding_features = ohe_encoding_features

        if le_encoding_features is None:
            self.le_encoding_features = ['user_agent.name']
        else:
            self.le_encoding_features = le_encoding_features

        if insufficent_columns is None:
            self.insufficent_columns = [
                '@timestamp', 'log.file.path', "message",
                "host.name", "url.original", "user_agent.original",
                "user_agent.version", "user_agent.device.name", "ecs.version",
                "event.dataset", "event.original", "source", "event.category",
                "http.version", "user_agent.os.version", "user_agent.os.full",
                "event.type", "event.kind", "event.module",
                "type", "timestamp", "timestamp_dt",
                "http.response.body.bytes", "source.ip"
            ]
        else:
            self.insufficent_columns = insufficent_columns

        if ohe_filepath is None or ohe_filepath == "":
            self.ohe_enc = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
        else:
            self.ohe_enc = joblib.load(ohe_filepath)

        if le_filepath is None or le_filepath == "":
            self.le_encoder = LabelEncoder()
        else:
            self.le_encoder = joblib.load(le_filepath)

    def learn_encoders(self, df: pd.DataFrame):
        """ Learn encoders for online answering """
        self.ohe_enc.fit(df[self.ohe_encoding_features])
        self.le_encoder = LabelEncoder().fit(df[self.le_encoding_features[0]].astype(str))

    @staticmethod
    def read_log_files(paths: list[str]) -> pd.DataFrame:
        """ Read log files on local machine """
        if len(paths) == 0:
            raise ValueError('Could not initialize data from empty logs')

        frames = []
        for path in paths:
            frames.append(
                pd.json_normalize(
                    pd.read_json(path, lines=True).to_dict("records"),
                    sep=".",
                )
            )
        return pd.concat(frames, axis=0, ignore_index=True)

    def ecs2pandas(self, data: str) -> pd.DataFrame:
        """ Gets the string in ecs format and returns it as a pandas dataframe, vectorized """
        return self.normalize(
            pd.json_normalize(
                json.loads(data), sep='.'
            )
        )

    def drop_insufficent_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Drops columns that have insufficient values """
        cols = [col for col in self.insufficent_columns if col in df.columns]
        if cols:
            df = df.drop(columns=cols)
        return df

    def encode_columns(self, df: pd.DataFrame) -> sparse.csr_matrix:
        """ Encodes columns """
        df_ohe = df[self.ohe_encoding_features].fillna("missing").copy()
        df_ohe = self._normalize_ohe_input(df_ohe)
        ohe_out = self.ohe_enc.transform(df_ohe)
        if not sparse.issparse(ohe_out):
            ohe_out = sparse.csr_matrix(ohe_out)

        le_sparse = []
        for col in self.le_encoding_features:
            col_data = self._normalize_le_input(df[col])
            le_vals = self.le_encoder.transform(col_data).reshape(-1, 1)
            le_sparse.append(sparse.csr_matrix(le_vals))

        if le_sparse:
            le_out = sparse.hstack(le_sparse, format="csr")
            return sparse.hstack([ohe_out, le_out], format="csr")
        return ohe_out

    def _normalize_ohe_input(self, df_ohe: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self.ohe_enc, "categories_"):
            return df_ohe

        for idx, col in enumerate(self.ohe_encoding_features):
            categories = self.ohe_enc.categories_[idx]
            if hasattr(categories, "dtype") and np.issubdtype(categories.dtype, np.number):
                cats_set = set(categories.tolist())
                default_val = categories[0] if len(categories) else 0
                series = pd.to_numeric(df_ohe[col], errors="coerce").fillna(default_val)
                series = series.where(series.isin(cats_set), default_val)
                df_ohe[col] = series
            else:
                cats_set = set([str(x) for x in categories])
                default_val = "missing" if "missing" in cats_set else (next(iter(cats_set)) if cats_set else "missing")
                series = df_ohe[col].astype(str)
                df_ohe[col] = series.where(series.isin(cats_set), default_val)
        return df_ohe

    def _normalize_le_input(self, series: pd.Series) -> pd.Series:
        if not hasattr(self.le_encoder, "classes_"):
            return series.fillna("missing").astype(str)
        classes = [str(x) for x in self.le_encoder.classes_]
        classes_set = set(classes)
        default_val = "missing" if "missing" in classes_set else (classes[0] if classes else "missing")
        data = series.fillna(default_val).astype(str)
        return data.where(data.isin(classes_set), default_val)

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Full pipeline of ECS vectorization ONLY HTTP COMPLETED"""
        df = df.copy()
        # Ensure source.ip exists (ES may provide source.address only)
        if 'source.ip' not in df.columns and 'source.address' in df.columns:
            df['source.ip'] = df['source.address']
        if 'url.path' not in df.columns:
            if 'request' in df.columns:
                df['url.path'] = df['request'].astype(str)
            elif 'url.original' in df.columns:
                df['url.path'] = df['url.original'].astype(str)
            else:
                raise ValueError("url.path missing")
        else:
            if 'url.original' in df.columns:
                df['url.path'] = df['url.path'].fillna(df['url.original'])

        df['url.path'] = df['url.path'].fillna("").astype(str)
        df['url.path'] = df['url.path'].str.split("?", n=1).str[0]

        ts_from_at = pd.Series([pd.NaT] * len(df))
        ts_from_ts = pd.Series([pd.NaT] * len(df))
        if '@timestamp' in df.columns:
            ts_from_at = pd.to_datetime(df['@timestamp'], errors='coerce', utc=True)
        if 'timestamp' in df.columns:
            ts_from_ts = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
            if ts_from_ts.isna().any():
                # Try Apache-style format as a fallback
                ts_fallback = pd.to_datetime(
                    df['timestamp'],
                    errors='coerce',
                    format="%d/%b/%Y:%H:%M:%S %z",
                )
                ts_from_ts = ts_from_ts.fillna(ts_fallback)

        df['timestamp_dt'] = ts_from_at.fillna(ts_from_ts)
        if df['timestamp_dt'].isna().any():
            logging.warning("timestamp parsing failed for some rows, using current time")
            df['timestamp_dt'] = df['timestamp_dt'].fillna(pd.Timestamp.now(tz="UTC"))

        df['day'] = df['timestamp_dt'].dt.day
        df['month'] = df['timestamp_dt'].dt.month
        df['hour'] = df['timestamp_dt'].dt.hour
        df['minute'] = df['timestamp_dt'].dt.minute
        df['http.response.body.bytes_log2'] = np.log2(
            pd.to_numeric(df['http.response.body.bytes'], errors='coerce').fillna(0) + 1
        )
        df = df.loc[df['url.path'].notna()]

        if 'source.ip' not in df.columns:
            raise ValueError("source.ip missing")

        ip_series = df['source.ip'].fillna('0.0.0.0').astype(str)
        df['ip_log2'] = np.log2(ip_series.map(_ip_to_int).astype(float) + 1.0)

        df = self.drop_insufficent_columns(df)

        df['url.file'] = df['url.path'].str.extract(r'/([^/]+\.[^/]+)$')
        df['url.file'] = df['url.file'].fillna('')

        df['url.directory'] = np.where(
            df['url.file'].notna(),
            df['url.path'].str.replace(r'/[^/]+$', '', regex=True),
            df['url.path']
        )

        df = df.drop('url.path', axis=1)

        encoded = self.encode_columns(df)
        drop_cols = [col for col in self.ohe_encoding_features + self.le_encoding_features if col in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        X_hashed, df = self.hash_http_data(df)
        non_numeric_cols = df.select_dtypes(include=['object']).columns
        if len(non_numeric_cols) > 0:
            df = df.drop(columns=non_numeric_cols)
        df = df.fillna(0)  # или SimpleImputer

        if df.empty:
            X_numeric = sparse.csr_matrix((0, 0))
        else:
            X_numeric = sparse.csr_matrix(df.to_numpy())

        return sparse.hstack([X_hashed, encoded, X_numeric], format="csr")

    def hash_http_data(self, df: pd.DataFrame) -> tuple[sparse.csr_matrix, pd.DataFrame]:
        X_dir = self.dir_vec.transform(df['url.directory'].fillna(''))
        X_file = self.file_vec.transform(df['url.file'].fillna(''))
        X_ref = self.ref_vec.transform(df['http.request.referrer'].fillna(''))

        X_hashed = sparse.hstack([X_dir, X_file, X_ref], format="csr")
        df = df.drop(['url.directory', 'url.file', 'http.request.referrer'], axis=1)
        return X_hashed, df

