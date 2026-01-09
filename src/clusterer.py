import joblib
import pandas as pd
from sklearn.cluster import HDBSCAN


class LogClustererModel:
    def __init__(self, model_filepath: str = None):
        if model_filepath is None or model_filepath == "":
            self.clusterer = HDBSCAN(
                min_cluster_size=50,
                min_samples=10,
                metric="euclidean"
            )
        else:
            self.clusterer = joblib.load(model_filepath)

    def learn_clusterer(self, data: pd.DataFrame):
        return self.clusterer.fit(data)

    def dump_model(self, model_filepath: str):
        joblib.dump(self.clusterer, model_filepath)

