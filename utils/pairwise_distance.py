from aeon.distances import (
    euclidean_pairwise_distance,
    dtw_pairwise_distance,
    twe_pairwise_distance,
    msm_pairwise_distance,
    wdtw_pairwise_distance,
    erp_pairwise_distance,
)

functions_dict = {
    "euclidean": euclidean_pairwise_distance,
    "dtw": dtw_pairwise_distance,
    "msm": msm_pairwise_distance,
    "twe": twe_pairwise_distance,
    "wdtw": wdtw_pairwise_distance,
    "erp": erp_pairwise_distance,
}
