import os
import time

import numpy as np


from utils.kmedoids import KMedoids
from utils.load_tsv import load_from_tsvfile
from utils.data_names import filtered_datanames


distance_vector = ["euclidean", "dtw", "msm", "twe", "wdtw", "erp"]

for dataname in filtered_datanames:
    print("dataset: ", dataname)
    trainpath = f"./data/{dataname}/{dataname}_TRAIN.tsv"
    testpath = f"./data/{dataname}/{dataname}_TEST.tsv"
    trainX, trainY = load_from_tsvfile(trainpath)
    testX, testY = load_from_tsvfile(testpath)

    Y = np.concatenate((trainY, testY))
    n_clust = len(np.unique(Y))

    t0 = time.time()

    for distance in distance_vector:
        print("distance: ", distance)
        try:
            train_predict, test_predict = KMedoids(distance, n_clust, trainX, testX)
            np.savetxt(
                f"./Results/{dataname}/Kmedoids/rawdata_kmedoid/Kmedoid_{distance}_train_predict.csv",
                train_predict,
                delimiter=",",
                fmt="%d",
            )
            np.savetxt(
                f"./Results/{dataname}/Kmedoids/rawdata_kmedoid/Kmedoid_{distance}_trainY.csv",
                trainY,
                delimiter=",",
                fmt="%d",
            )
            np.savetxt(
                f"./Results/{dataname}/Kmedoids/rawdata_kmedoid/Kmedoid_{distance}_test_predict.csv",
                test_predict,
                delimiter=",",
                fmt="%d",
            )
            np.savetxt(
                f"./Results/{dataname}/Kmedoids/rawdata_kmedoid/Kmedoid_{distance}_testY.csv",
                testY,
                delimiter=",",
                fmt="%d",
            )
        except Exception as e:
            print(e)
            continue

print(time.time() - t0)
