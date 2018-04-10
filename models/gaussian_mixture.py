import numpy as np
from sklearn import mixture
import pandas as pd
import utils.fraud_score as fs

def gaussian_mixture(normal, anormal, n_components, cov, tol, max_iter, reg_covar, type='Gaussian'):

    X = pd.concat([normal, anormal], axis=0)

    X_fraude = X[['id_siniestro', 'FRAUDE']]
    del X['FRAUDE']
    del X['id_siniestro']
    if type == 'Gaussian':
        db = mixture.GaussianMixture(n_components=n_components, covariance_type=cov, tol=tol,
                                 max_iter=max_iter, init_params='kmeans', random_state=541, reg_covar=reg_covar).fit(X)
    if type == 'Bayesian':
        db = mixture.BayesianGaussianMixture(n_components=n_components, covariance_type=cov, tol=tol,
                                             max_iter=max_iter, init_params='kmeans', random_state=541).fit(X)

    labels = db.predict(X)
    labels_df = pd.DataFrame(labels, index=X.index, columns=['Clusters'])
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
