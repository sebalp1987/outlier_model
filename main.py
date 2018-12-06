import os
import STRING
import pandas as pd
import models.model_evaluation as uns
from utils import train_test_utils
from models.sup_models import Models
from utils.model_utils import oversample_unsupervised
from models.mini_batch_kmeans import mini_batch_kmeans


def global_main():
    os.chdir(STRING.path_processing)

    normal = pd.read_csv('batch_files\\anomaly_raw_file_Dic17.csv', sep=';', encoding='latin1')
    anomaly = pd.read_csv('batch_files\\normal_raw_file_Dic17.csv', sep=';', encoding='latin1')

    normal = normal.drop_duplicates(subset='id_siniestro')
    anomaly = anomaly.drop_duplicates(subset='id_siniestro')

    # UNSUPERVISED MODEL SELECTION ####################################################################################

    # 1) First we calculate the Fraud Score for each model (you can try other commented models)
    oversample_times = None
    kmeans_fscore = uns.mini_batch_kmeans_tunning(normal, anomaly, oversample_times=oversample_times)
    # mix_fscore = uns.mixtures_tunning(normal, anomaly, oversample_times)
    # iforest_fscore = uns.isolation_forest_tunning(normal, anomaly, oversample_times)
    # dbscan_fscore = uns.dbscan_tunning(normal, anomaly, oversample_times)
    # hdbscan_fscore = uns.hdbscan_tunning(normal, anomaly, oversample_times)
    # svm_fscore = uns.super_vector_tunning(normal, anomaly, oversample_times)
    # ms_fscore = uns.mean_shift_tunning(normal, anomaly, oversample_times)
    # agglo_fscore = uns.agglomerative_tunning(normal, anomaly, oversample_times)

    # kmeans_fscore = pd.read_csv('final_files\\fs_kmeans_mini.csv', sep=';', encoding='latin1')
    # iforest_fscore = pd.read_csv('final_files\\fs_isolation_forest.csv', sep=';', encoding='latin1')
    # svm_fscore = pd.read_csv('final_files\\fs_svm.csv', sep=';', encoding='latin1')
    # dbscan_fscore = pd.read_csv('final_files\\fs_dbscan.csv', sep=';', encoding='latin1')
    # hdbscan_fscore = pd.read_csv('final_files\\fs_hdbscan.csv', sep=';', encoding='latin1')
    # ms_fscore = pd.read_csv('final_files\\fs_mean_shift.csv', sep=';', encoding='latin1')
    # agglo_fscore = pd.read_csv('final_files\\fs_agglomerative.csv', sep=';', encoding='latin1')
    # mix_fscore = pd.read_csv('final_files\\fs_mixture.csv', sep=';', encoding='latin1')

    # 2) We take the params that are related to the max Fraud Score
    kmeans_fscore = kmeans_fscore[kmeans_fscore['fraud_score'] == kmeans_fscore['fraud_score'].max()].reset_index(
        drop=True)

    # max_if = iforest_fscore.at[0, 'fraud_score']
    # max_if = 0
    #  max_svm = svm_fscore.at[0, 'fraud_score']
    # max_svm = 0
    # max_db = dbscan_fscore.at[0, 'fraud_score']
    # max_db = 0
    # max_hdb = hdbscan_fscore.at[0, 'fraud_score']
    # max_hdb = 0
    # max_ms = ms_fscore.at[0, 'fraud_score']
    # max_ms = 0
    # max_agglo = agglo_fscore.at[0, 'fraud_score']
    # max_agglo = 0
    # max_mix = mix_fscore[mix_fscore['fraud_score'] == mix_fscore['fraud_score'].max()].reset_index(drop=True)
    # max_mix = mix_fscore.at[0, 'fraud_score']
    # max_mix = 0

    '''
    # 3) We search which is the best clusterization model and keep the id_siniestro, Clusters variables
    if max_if >= max(max_svm, max_db, max_ms, max_kmeans, max_agglo, max_mix, max_hdb):
        contamination = iforest_fscore.at[0, 'contamination']
        n_estimators = iforest_fscore.at[0, 'n_estimators']
        from models.model_evaluation import isolation_forest
        print('Isolation Forest with ' + 'contamination ' + str(contamination) + ' n_estimators ' + str(n_estimators))
        _, _, _, comparative = isolation_forest(normal, anomaly, contamination=contamination, n_estimators=n_estimators)
    
    elif max_svm >= max(max_if, max_db, max_ms, max_kmeans, max_agglo, max_mix, max_hdb):
        nu = svm_fscore.at[0, 'nu']
        gamma = svm_fscore.at[0, 'gamma']
        from models.model_evaluation import super_vector
        print('SVM with ' + 'nu ' + str(nu) + ' gamma ' + str(gamma))
        _, _, _, comparative = super_vector(normal, anomaly, nu=nu, gamma=gamma)
    
    elif max_db >= max(max_if, max_svm, max_ms, max_kmeans, max_agglo, max_mix, max_hdb):
        eps = dbscan_fscore.at[0, 'eps']
        min_samples = dbscan_fscore.at[0, 'min_samples']
        from models.model_evaluation import dbscan
        print('DBSCAN with ' + 'eps ' + str(eps) + ' min_samples ' + str(min_samples))
        _, _, _, comparative = dbscan(normal, anomaly, eps=eps, min_samples=min_samples)
    
    elif max_hdb >= max(max_if, max_svm, max_ms, max_kmeans, max_agglo, max_mix, max_db):
        min_cluster_size = hdbscan_fscore.at[0, 'min_cluster_size']
        min_samples = hdbscan_fscore.at[0, 'min_samples']
        csm = hdbscan_fscore.at[0, 'csm']
        from models.model_evaluation import hdbscan
        print('HDBSCAN with ' + 'min_cluster ' + str(min_cluster_size) + ' min_samples ' + str(min_samples) + 'model ' +
              str(csm))
        _, _, _, comparative = hdbscan(normal, anomaly, min_cluster_size=min_cluster_size, min_samples=min_samples, csm=csm)
    
    elif max_ms >= max(max_if, max_svm, max_db, max_kmeans, max_agglo, max_mix, max_hdb):
        quantile = ms_fscore.at[0, 'quantile']
        from models.model_evaluation import mean_shift
        print('Mean Shift with ' + 'quantile ' + str(quantile))
        _, _, _, comparative = mean_shift(normal, anomaly, quantile=quantile)
    
    elif max_kmeans >= max(max_if, max_svm, max_db, max_ms, max_agglo, max_mix, max_hdb):
        max_iter = kmeans_fscore.at[0, 'max_iter']
        batch_size = kmeans_fscore.at[0, 'batch_size']
        n_clusters = kmeans_fscore.at[0, 'n_clusters']
    
        from models.model_evaluation import mini_batch_kmeans
        print('Mean Shift with ' + 'iters ' + str(max_iter) + 'batch ' + str(batch_size) + 'cluster ' + str(n_clusters))
        anomaly = anomaly.append([anomaly] * oversample_times, ignore_index=True)
        _, _, _, comparative = mini_batch_kmeans(normal, anomaly, n_clusters=n_clusters,
                                                 max_iter=max_iter, batch_size=batch_size)
    
        comparative = comparative.drop_duplicates(subset='id_siniestro')
        
    
    elif max_agglo >= max(max_if, max_svm, max_db, max_ms, max_kmeans, max_mix, max_hdb):
        linkage = agglo_fscore.at[0, 'linkage']
        connectivity = agglo_fscore.at[0, 'connectivity']
        n_clusters = agglo_fscore.at[0, 'n_clusters']
    
        from models.model_evaluation import agglomerative
        print('Agglomerative Clustering with ' + 'linkage ' + str(linkage) + 'connectivity '
              + str(connectivity) + 'cluster ' + str(n_clusters))
        _, _, _, comparative = agglomerative(normal, anomaly, n_clusters=n_clusters,
                                             linkage=linkage, connectivity=connectivity)
    
    elif max_mix >= max(max_if, max_svm, max_db, max_ms, max_kmeans, max_agglo, max_hdb):
        print(mix_fscore)
        n_components = mix_fscore.at[0, 'n_components']
        cov = mix_fscore.at[0, 'cov']
        model = mix_fscore.at[0, 'model']
        tol = mix_fscore.at[0, 'tol']
    
        from models.model_evaluation import gaussian_mixture
        print('Mixture with ' + 'n_components ' + str(n_components) + 'Covariance '
              + str(cov) + 'Model ' + str(model) + ' Tolerance ' + str(tol))
        anomaly = anomaly.append([anomaly] * oversample_times, ignore_index=True)
        _, _, _, comparative = gaussian_mixture(normal, anomaly, n_components=n_components, cov=cov, type=model,
                                                max_iter=500, reg_covar=1e-6, tol=tol)
    
     '''
    max_iter = kmeans_fscore.at[0, 'max_iter']
    batch_size = kmeans_fscore.at[0, 'batch_size']
    n_clusters = kmeans_fscore.at[0, 'n_clusters']

    print('Mean Shift with ' + 'iters ' + str(max_iter) + 'batch ' + str(batch_size) + 'cluster ' + str(n_clusters))
    if oversample_times is None:
        normal, anomaly = oversample_unsupervised(normal, anomaly)
    else:
        anomaly = anomaly.append([anomaly] * oversample_times, ignore_index=True)

    _, _, _, comparative = mini_batch_kmeans(normal, anomaly, n_clusters=n_clusters,
                                             max_iter=max_iter, batch_size=batch_size)

    comparative = comparative.drop_duplicates(subset='id_siniestro')
    comparative = comparative.drop_duplicates(subset=['id_siniestro', 'FRAUDE', 'Clusters', 'FRAUDE_Clusters'])
    comparative['id_siniestro'] = comparative['id_siniestro'].map(int)

    # 1) TRAIN, TEST: We separate without taking into account the Cluster but the FRAUD cases
    # Normal, Anormal, New
    normal = normal.drop_duplicates(subset='id_siniestro')
    anomaly = anomaly.drop_duplicates(subset='id_siniestro')

    # Train - Test - Valid
    train_t, test_t, valid_t = train_test_utils.train_test.training_test_valid(normal, anomaly)
    train_t['id_siniestro'] = train_t['id_siniestro'].map(int)
    test_t['id_siniestro'] = test_t['id_siniestro'].map(int)
    valid_t['id_siniestro'] = valid_t['id_siniestro'].map(int)
    train_t = pd.merge(train_t, comparative.drop(['FRAUDE'], axis=1), how='left', on='id_siniestro')
    test_t = pd.merge(test_t, comparative.drop(['FRAUDE'], axis=1), how='left', on='id_siniestro')
    valid_t = pd.merge(valid_t, comparative.drop(['FRAUDE'], axis=1), how='left', on='id_siniestro')
    train_t = train_t.drop_duplicates(subset='id_siniestro')
    test_t = test_t.drop_duplicates(subset='id_siniestro')
    valid_t = valid_t.drop_duplicates(subset='id_siniestro')

    # Labels
    labels = ['FRAUDE', 'Clusters', 'FRAUDE_Clusters']
    selected_label = 'FRAUDE_Clusters'
    labels.remove(selected_label)
    for i in labels:
        del train_t[i]
        del test_t[i]
        del valid_t[i]

    max_depth = 500
    oob_score = True
    class_weight = 'balanced_subsample'
    base_sampling = None
    control_sampling = 'ADASYN'
    bootstrap = True
    n_estimators = 500

    # 3) TUNNING THRESHOLD
    print('MODELO BASE THRESHOLD')
    Models.model_evaluation(train_t.copy(), valid_t.copy(), test_t.copy(), comparative.copy(),
                            bootstrap=bootstrap,
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            oob_score=oob_score,
                            class_weight=class_weight,
                            sampling=base_sampling,
                            label=selected_label, beta=2, model='ert')
    print('MODELO CONTROL THRESHOLD')
    Models.model_evaluation(train_t.copy(), valid_t.copy(), test_t.copy(),
                            comparative.copy(), bootstrap=bootstrap,
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            oob_score=oob_score,
                            class_weight=class_weight,
                            sampling=control_sampling,
                            label=selected_label, beta=2, model='ert')

    # OTROS MODELOS
    for model in ['gb', 'stacked_ERT', 'stacked_GB', 'stacked_LXGB', 'lxgb']:
        print('MODEL ' + model)
        Models.model_evaluation(train_t.copy(), valid_t.copy(), test_t.copy(),
                                comparative.copy(), bootstrap=bootstrap,
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                oob_score=oob_score,
                                class_weight=class_weight,
                                sampling=control_sampling,
                                label=selected_label, beta=2, model=model)
        print('MODEL ' + model + ' FINISHED')


if __name__ == '__main__':
    global_main()
