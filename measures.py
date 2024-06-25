from typing import List
import numpy as np
import pandas as pd

from tsml_eval.evaluation.metrics import clustering_accuracy_score
from sklearn.metrics.cluster import (
    adjusted_rand_score,
    rand_score,
    mutual_info_score,
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
)
from utils.load_tsv import load_from_tsvfile

column_names = [
    "distance",
    "accuracy",
    "adjusted rand score",
    "rand score",
    "mutual info score",
    "adjusted mutual info score",
    "normalised mutual info score",
]

def compute_measures(datanames: List[str], distances: List[str], datapath: str, resultpath: str, clusterer: str):
    """
    Clusterer should be either: "Kmedoid" or "Kmean"
    """
    for dataname in datanames:
        trainpath = f"./{datapath}/{dataname}/{dataname}_TRAIN.tsv"
        testpath = f"./{datapath}/{dataname}/{dataname}_TEST.tsv"
        trainX, trainY = load_from_tsvfile(trainpath)
        testX, testY = load_from_tsvfile(testpath)

        Y = np.concatenate((trainY, testY))
        n_clust = len(np.unique(Y))

        results_train = pd.DataFrame(columns=column_names)
        results_test = pd.DataFrame(columns=column_names)

        for distance in distances:
            train_predict = np.genfromtxt(
                f"./{resultpath}/{dataname}/Kmedoids/rawdata_kmedoid/{clusterer}_{distance}_train_predict.csv",
                delimiter=",",
            )
            test_predict = np.genfromtxt(
                f"./{resultpath}/{dataname}/Kmedoids/rawdata_kmedoid/{clusterer}_{distance}_trainY.csv",
                delimiter=",",
            )

            # Create DataFrame for train metrics
            new_row_train = pd.DataFrame(
                [
                    {
                        "distance": distance,
                        "accuracy": clustering_accuracy_score(train_predict, trainY),
                        "adjusted rand score": adjusted_rand_score(train_predict, trainY),
                        "rand score": rand_score(train_predict, trainY),
                        "mutual info score": mutual_info_score(train_predict, trainY),
                        "adjusted mutual info score": adjusted_mutual_info_score(
                            train_predict, trainY
                        ),
                        "normalised mutual info score": normalized_mutual_info_score(
                            train_predict, trainY
                        ),
                    }
                ]
            )

            # Create DataFrame for test metrics
            new_row_test = pd.DataFrame(
                [
                    {
                        "distance": distance,
                        "accuracy": clustering_accuracy_score(test_predict, testY),
                        "adjusted rand score": adjusted_rand_score(test_predict, testY),
                        "rand score": rand_score(test_predict, testY),
                        "mutual info score": mutual_info_score(test_predict, testY),
                        "adjusted mutual info score": adjusted_mutual_info_score(
                            test_predict, testY
                        ),
                        "normalised mutual info score": normalized_mutual_info_score(
                            test_predict, testY
                        ),
                    }
                ]
            )

            results_train = pd.concat([results_train, new_row_train], ignore_index=True)
            results_test = pd.concat([results_test, new_row_test], ignore_index=True)
        
        results_train.to_csv(f"./{resultpath}/measures_{datanames}.csv", index=False)
        results_test.to_csv(f"./{resultpath}/measures_{datanames}.csv", index=False)


        # print("train_results:\n",results_train)
        # print("train_results:\n",results_test)