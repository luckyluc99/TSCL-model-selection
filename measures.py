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
from utils.data_names import filtered_datanames

column_names = [
    "distance",
    "accuracy",
    "adjusted rand score",
    "rand score",
    "mutual info score",
    "adjusted mutual info score",
    "normalised mutual info score",
]

distance_vector = ["euclidean", "dtw", "msm", "twe", "wdtw", "erp"]

clusterer = "Kmedoid"  # "Kmean"

for dataname in filtered_datanames:
    trainpath = f"./data/{dataname}/{dataname}_TRAIN.tsv"
    testpath = f"./data/{dataname}/{dataname}_TEST.tsv"
    trainX, trainY = load_from_tsvfile(trainpath)
    testX, testY = load_from_tsvfile(testpath)

    Y = np.concatenate((trainY, testY))
    n_clust = len(np.unique(Y))

    results_train = pd.DataFrame(columns=column_names)
    results_test = pd.DataFrame(columns=column_names)

    for distance in distance_vector:
        train_predict = np.genfromtxt(
            f".Results/{dataname}/Kmedoids/rawdata_kmedoid/{clusterer}_{distance}_train_predict.csv",
            delimiter=",",
        )
        test_predict = np.genfromtxt(
            f"./Results/{dataname}/Kmedoids/rawdata_kmedoid/{clusterer}_{distance}_trainY.csv",
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

    resultpath_train = f"./Modelselection_clustering/{dataname}/{clusterer}_{dataname}_output_train.csv"
    resultpath_test = (
        f"./Modelselection_clustering/{dataname}/{clusterer}_{dataname}_output_test.csv"
    )

    results_train.to_csv(resultpath_train, index=False)
    results_test.to_csv(resultpath_test, index=False)
