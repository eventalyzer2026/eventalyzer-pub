import ipaddress
import json
import logging

import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


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
            self.ohe_enc = OneHotEncoder(sparse_output=False)
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

        df = pd.DataFrame()

        for path in paths:
            df = pd.concat([df.copy(), pd.json_normalize(pd.read_json(path, lines=True).to_dict("records"), sep=".")],
                           axis=1)

        return df

    def ecs2pandas(self, data: str) -> pd.DataFrame:
        """ Gets the string in ecs format and returns it as a pandas dataframe, vectorized """
        return self.normalize(
            pd.json_normalize(
                json.loads(data), sep='.'
            )
        )

    def drop_insufficent_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Drops columns that have insufficient values """
        # logging.info(df.columns)
        for col in self.insufficent_columns:
            try:
                df = df.drop(col, axis=1)
            except KeyError:
                continue
                # logging.info("Could not drop column: %s", col)

        return df

    def encode_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Encodes columns """
        df_ohe = df[self.ohe_encoding_features].fillna("missing")
        ohe_df = pd.DataFrame(
            self.ohe_enc.transform(df_ohe),
            columns=self.ohe_enc.get_feature_names_out(self.ohe_encoding_features)
        )

        le_dfs = []
        for col in self.le_encoding_features:
            col_data = df[col].fillna("missing").astype(str)
            le_dfs.append(pd.DataFrame(self.le_encoder.transform(col_data), columns=[col + "_le"]))
        le_df = pd.concat(le_dfs, axis=1)

        return pd.concat([ohe_df, le_df], axis=1)

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Full pipeline of ECS vectorization ONLY HTTP COMPLETED"""
        df = df.copy()
        ts_col = '@timestamp' if 'timestamp' in df.columns else 'timestamp'
        df['timestamp_dt'] = pd.to_datetime(df[ts_col], errors='coerce')
        if df['timestamp_dt'].isna().any():
            raise ValueError("timestamp parsing failed")

        df['day'] = df['timestamp_dt'].dt.day
        df['month'] = df['timestamp_dt'].dt.month
        df['hour'] = df['timestamp_dt'].dt.hour
        df['minute'] = df['timestamp_dt'].dt.minute
        df['http.response.body.bytes_log2'] = np.log2(df['http.response.body.bytes'].fillna(0) + 1)
        df = df[df['url.path'].notna()]

        df['ip_log2'] = np.log2(
            df['source.ip']
            .fillna('0.0.0.0')
            .map(lambda x: float(int(ipaddress.ip_address(x))))
            + 1.0
        )

        df = self.drop_insufficent_columns(df)

        df['url.file'] = df['url.path'].str.extract(r'/([^/]+\.[^/]+)$')
        df['url.file'] = df['url.file'].fillna('')

        df['url.directory'] = np.where(
            df['url.file'].notna(),
            df['url.path'].str.replace(r'/[^/]+$', '', regex=True),
            df['url.path']
        )

        df = df.drop('url.path', axis=1)

        df = pd.concat([self.encode_columns(df), df], axis=1)
        df.drop(self.ohe_encoding_features, axis=1, inplace=True)
        df.drop(self.le_encoding_features, axis=1, inplace=True)

        df = self.hash_http_data(df)
        df = df.fillna(0)  # или SimpleImputer

        # logging.info(f"Found {df.isna().sum().sum()} NaNs", )

        return df

    def hash_http_data(self, df: pd.DataFrame) -> pd.DataFrame:
        X_dir = self.dir_vec.transform(df['url.directory'].fillna(''))
        X_file = self.file_vec.transform(df['url.file'].fillna(''))
        X_ref = self.ref_vec.transform(df['http.request.referrer'].fillna(''))

        df_hashed = [
            pd.DataFrame.sparse.from_spmatrix(
                x, columns=[f'{['directory', 'file', 'ref'][j]}_feature_{i}' for i in range(x.shape[1])]
            ) for j, x in enumerate([X_dir, X_file, X_ref])
        ]

        df_hashed.append(df.drop(['url.directory', 'url.file', 'http.request.referrer'], axis=1))
        data = pd.concat(df_hashed, axis=1)

        return data

    def dump_vectorizer(self):
        pass
