import os
import time

import numpy as np
import pandas as pd

from utils.kmeans import KMeans
from utils.load_tsv import load_from_tsvfile
from utils.data_names import filtered_datanames

distances = ["euclidean", "dtw", "msm", "twe", "wdtw", "erp"]

t0 = time.time()
for dataname in filtered_datanames:
    print("dataset: ", dataname)
    trainpath = f"./data/{dataname}/{dataname}_TRAIN.tsv"
    testpath = f"./data/{dataname}/{dataname}_TEST.tsv"
    trainX, trainY = load_from_tsvfile(trainpath)
    testX, testY = load_from_tsvfile(testpath)

    Y = np.concatenate((trainY, testY))
    n_clust = len(np.unique(Y))

    for distance in distances:
        print("distance: ", distance)
        try:
            train_predict, test_predict = KMeans(distance, n_clust, trainX, testX)
            np.savetxt(
                f"./Results/{dataname}/rawdata_kmeans/Kmean_{distance}_train_predict.csv",
                train_predict,
                delimiter=",",
                fmt="%d",
            )
            np.savetxt(
                f"./Results/{dataname}/rawdata_kmeans/Kmean_{distance}_trainY.csv",
                trainY,
                delimiter=",",
                fmt="%d",
            )
            np.savetxt(
                f"./Results/{dataname}/rawdata_kmeans/Kmean_{distance}_test_predict.csv",
                test_predict,
                delimiter=",",
                fmt="%d",
            )
            np.savetxt(
                f"./Results/{dataname}/rawdata_kmeans/Kmean_{distance}_testY.csv",
                testY,
                delimiter=",",
                fmt="%d",
            )
        except Exception as e:
            print(e)
            continue

print(time.time() - t0)
