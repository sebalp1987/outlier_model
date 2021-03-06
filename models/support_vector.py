from sklearn import svm
import pandas as pd
import numpy as np
import utils.fraud_score as fs

def super_vector(normal, anormal, nu = 0.1, gamma = 0.1):

    X = pd.concat([normal, anormal], axis=0)

    X_fraude = X[['id_siniestro', 'FRAUDE']]
    del X['FRAUDE']
    del X['id_siniestro']


    db = svm.OneClassSVM(nu=nu, kernel="linear", gamma=gamma)
    db.fit(X)


    labels = db.predict(X)
    labels_df = pd.DataFrame(labels, index = X.index, columns=['Clusters'])
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    comparative = pd.concat([X_fraude, labels_df], axis=1)
    f1, f2, fscore, df_clusters = fs.fraud_score(comparative.drop(['id_siniestro'], axis=1), 'FRAUDE', 'Clusters')
    comparative['FRAUDE_Clusters'] = pd.Series(0, index=comparative.index)
    comparative['FRAUDE'] = comparative['FRAUDE'].map(int)
    comparative.loc[comparative['FRAUDE'] == 1, 'FRAUDE_Clusters'] = 1
    comparative.loc[comparative['Clusters'].isin(df_clusters), 'FRAUDE_Clusters'] = 1
    return f1, f2, fscore, comparative
