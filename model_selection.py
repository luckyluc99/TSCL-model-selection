from typing import List
import numpy as np
import pandas as pd

from sklearn_extra.cluster import KMedoids
from sklearn.metrics import (
    davies_bouldin_score,
    silhouette_score,
    calinski_harabasz_score,
)
from utils.pairwise_distance import functions_dict
from utils.load_tsv import load_from_tsvfile

distances = ["euclidean", "dtw", "msm", "twe", "wdtw", "erp"]

def model_selection(datanames: List[str], distances: List[str], datapath: str, resultpath: str):
    """
    Performs model selection using Davies-Bouldin, Silhouette and Calinksi-Harabasz indices.
    Datanames should be a list of the UCR time series archive names
    Distances should be of["euclidean", "dtw", "msm", "twe", "wdtw", "erp"]
    """
    results_silhouette = pd.DataFrame(columns=["datanames", "k"])
    results_davies_bouldin = pd.DataFrame(columns=["datanames", "k"])
    results_calinski_harabasz = pd.DataFrame(columns=["datanames", "k"])

    for distance in distances:
        print("distance: ", distance)
        for dataname in datanames:
            print("dataset: ", dataname)
            trainpath = f"./{datapath}/{dataname}/{dataname}_TRAIN.tsv"
            testpath = f"./{datapath}/{dataname}/{dataname}_TEST.tsv"
            trainX, trainY = load_from_tsvfile(trainpath)
            testX, testY = load_from_tsvfile(testpath)

            # For experiments on merged training and test sets
            X = np.concatenate((trainX, testX))
            Y = np.concatenate((trainY, testY))

            pairwise_distance = functions_dict[distance](trainX)

            k_opt = len(np.unique(trainY))
            k_min = np.maximum(2, k_opt - 10)
            k_max = np.minimum(k_opt + 10, np.shape(trainX)[0] - 1)

            silhouette_scores = {}
            davies_bouldin_scores = {}
            calinski_harabasz_scores = {}

            for k in range(k_min, k_max + 1):
                kmedoids = KMedoids(n_clusters=k, metric="precomputed").fit(
                    pairwise_distance
                )
                Y_predict = kmedoids.labels_

                if len(set(Y_predict)) != 1:
                    silhouette_scores[f"{k}"] = silhouette_score(
                        X=pairwise_distance, labels=Y_predict, metric="precomputed"
                    )
                    davies_bouldin_scores[f"{k}"] = davies_bouldin_score(
                        np.squeeze(trainX), Y_predict
                    )
                    calinski_harabasz_scores[f"{k}"] = calinski_harabasz_score(
                        np.squeeze(trainX), Y_predict
                    )

            if silhouette_scores:
                best_silhouette_k = max(silhouette_scores, key=silhouette_scores.get)
                new_silhouette = pd.DataFrame(
                    [{"datanames": dataname, "k": best_silhouette_k}]
                )
                results_silhouette = pd.concat(
                    [results_silhouette, new_silhouette], ignore_index=True
                )

            if davies_bouldin_scores:
                best_db_k = min(davies_bouldin_scores, key=davies_bouldin_scores.get)
                new_davies_bouldin = pd.DataFrame([{"datanames": dataname, "k": best_db_k}])
                results_davies_bouldin = pd.concat(
                    [results_davies_bouldin, new_davies_bouldin], ignore_index=True
                )

            if calinski_harabasz_scores:
                best_ch_k = max(calinski_harabasz_scores, key=calinski_harabasz_scores.get)
                new_calinski_harabasz = pd.DataFrame(
                    [{"datanames": dataname, "k": best_ch_k}]
                )
                results_calinski_harabasz = pd.concat(
                    [results_calinski_harabasz, new_calinski_harabasz], ignore_index=True
                )

        results_silhouette.to_csv(f"./{resultpath}/silhouette_{distance}.csv", index=False)
        results_davies_bouldin.to_csv(f"./{resultpath}/db_{distance}.csv", index=False)
        results_calinski_harabasz.to_csv(f"./{resultpath}/ch_{distance}.csv", index=False)
