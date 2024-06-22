from aeon.clustering import TimeSeriesKMedoids


def KMedoids(distance, nclusters, train_X, test_X):
    clst = TimeSeriesKMedoids(
        distance=distance,
        n_clusters=nclusters,
        init_algorithm="random",
        n_init=10,
        random_state=1,
    )

    clst.fit(train_X)
    train_predict = clst.predict(train_X)
    test_predict = clst.predict(test_X)
    return train_predict, test_predict
