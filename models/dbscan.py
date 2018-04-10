import numpy as np
from sklearn.cluster import DBSCAN
import pandas as pd
import utils.fraud_score as fs

def dbscan(normal, anormal, eps=0.3, min_samples=175, leaf_size=30):

    X = pd.concat([normal, anormal], axis=0)

    X_fraude = X[['id_siniestro', 'FRAUDE']]
    del X['FRAUDE']
    del X['id_siniestro']

    db = DBSCAN(eps=eps, min_samples=min_samples, leaf_size=leaf_size, n_jobs=-1).fit(X)

    labels = db.labels_
    labels_df = pd.DataFrame(labels, index = X.index, columns=['Clusters'])
    labels = db.labels_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print(n_clusters_)

    comparative = pd.concat([X_fraude, labels_df], axis=1)
    f1, f2, fscore, df_clusters = fs.fraud_score(comparative.drop(['id_siniestro'], axis=1), 'FRAUDE', 'Clusters')
    comparative['FRAUDE_Clusters'] = pd.Series(0, index=comparative.index)
    comparative['FRAUDE'] = comparative['FRAUDE'].map(int)
    comparative.loc[comparative['FRAUDE'] == 1, 'FRAUDE_Clusters'] = 1
    comparative.loc[comparative['Clusters'].isin(df_clusters), 'FRAUDE_Clusters'] = 1

    return f1, f2, fscore, comparative


