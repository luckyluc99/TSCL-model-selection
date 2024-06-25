import numpy as np

from typing import List
from utils.kmeans import KMeans
from utils.kmedoids import KMedoids
from utils.load_tsv import load_from_tsvfile


distance_vector = ["euclidean", "dtw", "msm", "twe", "wdtw", "erp"]

def compute_kmeans(datanames: List[str], distances: List[str], datapath: str, resultpath: str):
    """
    Computes the labels of both train and test sets using the k means clusterer.
    Writes the results to specified result path.
    Distances should list of: "euclidean", "dtw", "msm", "twe", "wdtw", "erp"
    Datanames should be list of the datasets of the UCR archive, however also other datasets could be used.
    """ 
    for dataname in datanames:
        print("dataset: ", dataname)
        trainpath = f"./{datapath}/{dataname}/{dataname}_TRAIN.tsv"
        testpath = f"./{datapath}/{dataname}/{dataname}_TEST.tsv"
        trainX, trainY = load_from_tsvfile(trainpath)
        testX, testY = load_from_tsvfile(testpath)

        Y = np.concatenate((trainY, testY))
        n_clust = len(np.unique(Y))

        for distance in distances:
            print("Distance: ", distance)
            try:
                train_predict, test_predict = KMeans(distance, n_clust, trainX, testX)
                np.savetxt(
                    f"./{resultpath}/rawdata_kmedoid/Kmean_{distance}_train_predict.csv",
                    train_predict,
                    delimiter=",",
                    fmt="%d",
                )
                np.savetxt(
                    f"./{resultpath}/rawdata_kmeans/Kmean_{distance}_trainY.csv",
                    trainY,
                    delimiter=",",
                    fmt="%d",
                )
                np.savetxt(
                    f"./{resultpath}/rawdata_kmedoid/Kmean_{distance}_test_predict.csv",
                    test_predict,
                    delimiter=",",
                    fmt="%d",
                )
                np.savetxt(
                    f"./{resultpath}/rawdata_kmedoid/Kmean_{distance}_testY.csv",
                    testY,
                    delimiter=",",
                    fmt="%d",
                )
            except Exception as e:
                print(e)
                continue

def compute_kmedoids(datanames: List[str], distances: List[str], datapath: str, resultpath: str):
    """
    Computes the labels of both train and test sets using the k medoids clusterer.
    Writes the results to specified result path.
    Distances should list of: "euclidean", "dtw", "msm", "twe", "wdtw", "erp"
    Datanames should be list of the datasets of the UCR archive, however also other datasets could be used.
    """ 
    for dataname in datanames:
        print("dataset: ", dataname)
        trainpath = f"./{datapath}/{dataname}/{dataname}_TRAIN.tsv"
        testpath = f"./{datapath}/{dataname}/{dataname}_TEST.tsv"
        trainX, trainY = load_from_tsvfile(trainpath)
        testX, testY = load_from_tsvfile(testpath)

        Y = np.concatenate((trainY, testY))
        n_clust = len(np.unique(Y))

        for distance in distances:
            print("Distance: ", distance)
            try:
                train_predict, test_predict = KMedoids(distance, n_clust, trainX, testX)
                np.savetxt(
                    f"./{resultpath}/rawdata_kmedoid/Kmedoid_{distance}_train_predict.csv",
                    train_predict,
                    delimiter=",",
                    fmt="%d",
                )
                np.savetxt(
                    f"./{resultpath}/rawdata_kmedoid/Kmedoid_{distance}_trainY.csv",
                    trainY,
                    delimiter=",",
                    fmt="%d",
                )
                np.savetxt(
                    f"./{resultpath}/rawdata_kmedoid/Kmedoid_{distance}_test_predict.csv",
                    test_predict,
                    delimiter=",",
                    fmt="%d",
                )
                np.savetxt(
                    f"./{resultpath}/rawdata_kmedoid/Kmedoid_{distance}_testY.csv",
                    testY,
                    delimiter=",",
                    fmt="%d",
                )
            except Exception as e:
                print(e)
                continue
