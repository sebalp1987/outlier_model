import utils.fraud_score as fs
from sklearn.cluster import MeanShift, estimate_bandwidth
import pandas as pd
import numpy as np

def mean_shift(normal, anormal, quantile = 0.5):

    X = pd.concat([normal, anormal], axis=0)
    X_fraude = X[['id_siniestro', 'FRAUDE']]
    del X['FRAUDE']
    del X['id_siniestro']

    bandwith = estimate_bandwidth(X.values, quantile=quantile, random_state= 42)

    db = MeanShift(bandwidth=bandwith, bin_seeding= True, cluster_all=False,
                   min_bin_freq=50, n_jobs=-1).fit(X)

    labels = db.labels_
    labels_df = pd.DataFrame(labels, index = X.index, columns=['Clusters'])
    labels = db.labels_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    cluster_centers = db.cluster_centers_


    comparative = pd.concat([X_fraude, labels_df], axis=1)
    f1, f2, fscore, df_clusters = fs.fraud_score(comparative.drop(['id_siniestro'], axis=1), 'FRAUDE', 'Clusters')
    comparative['FRAUDE_Clusters'] = pd.Series(0, index=comparative.index)
    comparative['FRAUDE'] = comparative['FRAUDE'].map(int)
    comparative.loc[comparative['FRAUDE'] == 1, 'FRAUDE_Clusters'] = 1
    comparative.loc[comparative['Clusters'].isin(df_clusters), 'FRAUDE_Clusters'] = 1
    return f1, f2, fscore, comparative
