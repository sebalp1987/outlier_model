import pandas as pd
from .isolation_forest import isolation_forest
from .support_vector import super_vector
from .dbscan import dbscan
# from .hd_bscan import hdbscan
from .mean_shift import mean_shift
from .mini_batch_kmeans import mini_batch_kmeans
from .agglomerative_cluster import agglomerative
from .gaussian_mixture import gaussian_mixture
import numpy as np
import os
import time
import STRING
from utils.model_utils import oversample_unsupervised

'''
# FILES
# Load File
normal = ReadCsv.load_csv(STRING.path_final_files + 'normal_raw_file.csv')
anormal = ReadCsv.load_csv(STRING.path_final_files + 'anomaly_raw_file.csv')

# Transform file to DF
normal_df = DfUtils.processing_file(normal)
anormal_df = DfUtils.processing_file(anormal)

# LABEL PROPAGATION

label_spreading(normal_df, anormal_df, kernel='rbf', gamma=20, n_neighbors=20, alpha=0.2, max_iter=30, tol=1e-3)
label_propagation(normal_df, anormal_df, kernel='rbf', gamma=20, n_neighbors=20, alpha=0.2, max_iter=30, tol=1e-3)
'''

# ISOLATION FOREST
def isolation_forest_tunning(normal_df, anormal_df, oversample_times, max_cont=0.2, max_n_estim=301):

    fraud_list_score = []
    i = 0
    anormal_df = anormal_df.append([anormal_df] * oversample_times, ignore_index=True)
    for contamination in np.arange(0.005, max_cont, 0.005):
        for n_estimators in np.arange(100, max_n_estim, 100):
            i += 1
            print(i)
            t_0 = time.time()
            f1, f2, fraud_score, _ = isolation_forest(normal_df, anormal_df, contamination=contamination, n_estimators=
            n_estimators)
            print(time.time() - t_0)
            fraud_list_score.append([contamination, n_estimators, f1, f2, fraud_score])

    fraud_list_score = pd.DataFrame(fraud_list_score, columns=['contamination', 'n_estimators', 'f1', 'f2', 'fraud_score'])
    max_fraud_score = fraud_list_score[fraud_list_score['fraud_score'] == fraud_list_score['fraud_score'].max()]
    print(fraud_list_score)
    print('max value ', max_fraud_score)
    df = pd.DataFrame(fraud_list_score, columns=['contamination', 'n_estimators', 'f1', 'f2', 'fraud_score'])
    df.to_csv('final_files\\fs_isolation_forest.csv', sep=';', encoding='latin1', index=False)

    return max_fraud_score



# ONE CLASS SVM
def super_vector_tunning(normal_df, anormal_df, oversample_times, max_nu=1, max_gamma=1):
    fraud_list_score = []
    i = 0
    anormal_df = anormal_df.append([anormal_df] * oversample_times, ignore_index=True)
    for nu in np.arange(0.1, max_nu, 0.1):
        for gamma in np.arange(0.1, max_gamma, 0.1):
            i += 1
            print(i)
            f1, f2, fraud_score, _ = super_vector(normal_df, anormal_df, nu=nu, gamma=gamma)
            fraud_list_score.append([nu, gamma, f1, f2, fraud_score])

    fraud_list_score = pd.DataFrame(fraud_list_score,
                                    columns=['nu', 'gamma', 'f1', 'f2', 'fraud_score'])
    max_fraud_score = fraud_list_score[fraud_list_score['fraud_score'] == fraud_list_score['fraud_score'].max()]
    print(fraud_list_score)
    print('max value ', max_fraud_score)
    df = pd.DataFrame(fraud_list_score,  columns=['nu', 'gamma', 'f1', 'f2', 'fraud_score'])
    df.to_csv('final_files\\fs_svm.csv', sep=';', encoding='latin1', index=False)

    return max_fraud_score


# DBSCAN
def dbscan_tunning(normal_df, anormal_df, oversample_times, max_eps=1.01, min_samples_max=3001):
    fraud_list_score = []
    i = 0
    anormal_df = anormal_df.append([anormal_df] * oversample_times, ignore_index=True)
    for eps in np.arange(0.1, max_eps, 0.1):
        for min_samples in np.arange(1000, min_samples_max, 1000):
            i += 1
            print(i)
            t_0 =time.time()
            f1, f2, fraud_score, _ = dbscan(normal_df, anormal_df, eps=eps, min_samples=min_samples,
                                         leaf_size=min_samples*0.1)
            print(time.time() - t_0)
            fraud_list_score.append([eps, min_samples, f1, f2, fraud_score])

    fraud_list_score = pd.DataFrame(fraud_list_score,
                                    columns=['eps', 'min_samples', 'f1', 'f2', 'fraud_score'])
    max_fraud_score = fraud_list_score[fraud_list_score['fraud_score'] == fraud_list_score['fraud_score'].max()]
    print(fraud_list_score)
    print('max value ', max_fraud_score)
    df = pd.DataFrame(fraud_list_score, columns=['eps', 'min_samples', 'f1', 'f2', 'fraud_score'])
    df.to_csv('final_files\\fs_dbscan.csv', sep=';', encoding='latin1', index=False)

    return max_fraud_score

# HDBSCAN
def hdbscan_tunning(normal_df, anormal_df, oversample_times, min_samples_max=3001, min_cluster_size_max=30001):
    fraud_list_score = []
    i = 0
    anormal_df = anormal_df.append([anormal_df] * oversample_times, ignore_index=True)
    for csm in ['eom', 'leaf']:
        for min_samples in np.arange(1000, min_samples_max, 1000):
            for min_cluster_size in np.arange(10000, min_cluster_size_max, 10000):
                i += 1
                print(i)
                f1, f2, fraud_score, _ = hdbscan(normal_df, anormal_df, csm=csm, min_samples=int(min_samples),
                                             min_cluster_size=int(min_cluster_size))
                fraud_list_score.append([csm, min_samples, min_cluster_size, f1, f2, fraud_score])

    fraud_list_score = pd.DataFrame(fraud_list_score,
                                    columns=['csm', 'min_samples', 'min_cluster_size', 'f1', 'f2', 'fraud_score'])
    max_fraud_score = fraud_list_score[fraud_list_score['fraud_score'] == fraud_list_score['fraud_score'].max()]
    print(fraud_list_score)
    print('max value ', max_fraud_score)
    df = pd.DataFrame(fraud_list_score, columns=['csm', 'min_samples', 'min_cluster_size', 'f1', 'f2', 'fraud_score'])
    df.to_csv('final_files\\fs_hdbscan.csv', sep=';', encoding='latin1', index=False)

    return max_fraud_score

# MEAN SHIFT
def mean_shift_tunning(normal_df, anormal_df, oversample_times, max_quantile=0.99):
    fraud_list_score = []
    i = 0
    anormal_df = anormal_df.append([anormal_df] * oversample_times, ignore_index=True)
    for quantile in np.arange(0.1, max_quantile, 0.05):
        i += 1
        print(i)
        f1, f2, fraud_score, _ = mean_shift(normal_df, anormal_df, quantile=quantile)
        fraud_list_score.append([quantile, f1, f2, fraud_score])

    fraud_list_score = pd.DataFrame(fraud_list_score,
                                    columns=['quantile','f1', 'f2', 'fraud_score'])
    max_fraud_score = fraud_list_score[fraud_list_score['fraud_score'] == fraud_list_score['fraud_score'].max()]
    print(fraud_list_score)
    print('max fraud score ', max_fraud_score)
    df = pd.DataFrame(fraud_list_score, columns=['quantile','f1', 'f2', 'fraud_score'])
    df.to_csv('final_files\\fs_mean_shift.csv', sep=';', encoding='latin1', index=False)

    return max_fraud_score


# MINI BATCH KMEANS
def mini_batch_kmeans_tunning(normal_df: pd.DataFrame, anormal_df: pd.DataFrame, oversample_times, max_iter=5001,
                              batch_size=10001, n_clusters=10):
    if oversample_times is None:
        normal_df, anormal_df = oversample_unsupervised(normal_df, anormal_df)

    else:
        anormal_df = anormal_df.append([anormal_df] * oversample_times, ignore_index=True)
    fraud_list_score = []
    i = 0
    for iter in np.arange(100, max_iter, 100):
        for batch_size in np.arange(100, batch_size, 100):
            for n_clusters in np.arange(2, n_clusters, 1):
                i += 1
                print(i)
                f1, f2, fraud_score, _ = mini_batch_kmeans(normal_df, anormal_df, max_iter=iter,
                                                           batch_size=batch_size, n_clusters=n_clusters)

                fraud_list_score.append([max_iter, batch_size, n_clusters, f1, f2, fraud_score])
    print(fraud_list_score)
    fraud_list_score = pd.DataFrame(fraud_list_score,
                                    columns=['max_iter', 'batch_size', 'n_clusters', 'f1', 'f2', 'fraud_score'])
    max_fraud_score = fraud_list_score[fraud_list_score['fraud_score'] == fraud_list_score['fraud_score'].max()]

    print('max fraud score ', max_fraud_score)
    df = pd.DataFrame(fraud_list_score, columns=['max_iter', 'batch_size', 'n_clusters', 'f1', 'f2', 'fraud_score'])
    import datetime
    DAY = datetime.datetime.today().strftime('%Y-%m-%d')
    path_no_supervisado = 'final_files\\' + str(DAY) + '\\no_supervisado\\'
    os.makedirs(os.path.dirname(path_no_supervisado), exist_ok=True)
    df.to_csv(path_no_supervisado + 'fs_kmeans_mini.csv', sep=';', encoding='latin1', index=False)

    return max_fraud_score

# AGGLOMERATIVE CLUSTERING
def agglomerative_tunning(normal_df, oversample_times, anormal_df, n_clusters = 10):
    fraud_list_score = []
    i = 0
    anormal_df = anormal_df.append([anormal_df] * oversample_times, ignore_index=True)
    from sklearn.neighbors import kneighbors_graph

    X = pd.concat([normal_df, anormal_df], axis=0)
    del X['FRAUDE']
    del X['id_siniestro']

    knn_graph = kneighbors_graph(X, 4, mode='distance', include_self=False, n_jobs=-1)

    for connectivity in (None, knn_graph):
        for linkage in ['average', 'complete', 'ward']:
            for n_clusters in np.arange(2, n_clusters, 1):
                i += 1
                print(i)
                f1, f2, fraud_score, _ = agglomerative(normal_df, anormal_df, linkage=linkage,
                                                       n_clusters=n_clusters, connectivity=connectivity)
                fraud_list_score.append([connectivity, linkage, n_clusters, f1, f2, fraud_score])
    print(fraud_list_score)
    fraud_list_score = pd.DataFrame(fraud_list_score,
                                    columns=['connectivity', 'linkage', 'n_clusters', 'f1', 'f2', 'fraud_score'])
    max_fraud_score = fraud_list_score[fraud_list_score['fraud_score'] == fraud_list_score['fraud_score'].max()]
    print('max fraud score ', max_fraud_score)
    df = pd.DataFrame(fraud_list_score, columns=['connectivity', 'linkage', 'n_clusters', 'f1', 'f2', 'fraud_score'])
    df.to_csv('final_files\\fs_agglomerative.csv', sep=';', encoding='latin1', index=False)


# MIXTURES
def mixtures_tunning(normal_df, anormal_df, oversample_times, max_components=7, max_tol=0.3):
    fraud_list_score = []
    i = 0
    anormal_df = anormal_df.append([anormal_df] * oversample_times, ignore_index=True)
    for n_components in np.arange(2, max_components, 1):
        for cov in ['spherical', 'diag', 'tied', 'full']:
            for model in ['Gaussian', 'Bayesian']:
                for tol in np.arange(0.01, max_tol, 0.02):
                    i += 1
                    print(i)
                    t_0 = time.time()
                    f1, f2, fraud_score, _ = gaussian_mixture(normal_df, anormal_df,
                                                              n_components, cov, tol=tol, max_iter=500, reg_covar=1e-6,
                                                              type=model)
                    print(time.time() - t_0)
                    fraud_list_score.append([n_components, cov, model, tol, f1, f2, fraud_score])
    print(fraud_list_score)
    fraud_list_score = pd.DataFrame(fraud_list_score,
                                    columns=['n_components', 'cov', 'model', 'tol', 'f1', 'f2', 'fraud_score'])
    max_fraud_score = fraud_list_score[fraud_list_score['fraud_score'] == fraud_list_score['fraud_score'].max()]
    print('max fraud score ', max_fraud_score)
    df = pd.DataFrame(fraud_list_score, columns=['n_components', 'cov', 'model', 'tol', 'f1', 'f2', 'fraud_score'])
    df.to_csv('final_files\\fs_mixture.csv', sep=';', encoding='latin1', index=False)

    return max_fraud_score