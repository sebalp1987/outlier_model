import os
import STRING
import pandas as pd
from utils.process_utils import process_utils
from models.model_evaluation import mini_batch_kmeans
import models.model_evaluation as uns
from utils import train_test_utils
from models.extreme_random_tree import extreme_randomize as ert
# import lightgbm as lgboost
# from mlxtend.classifier import StackingClassifier
# import logging
# from sklearn import ensemble, neighbors


def global_main():
    '-----------------------------------------PROCESSING TEST-------------------------------------------------------------'
    # Aquí están todas las botellas de test (menos BL y mediador que se calculan semanalmente)
    os.chdir(STRING.path_processing)
    df_test = pd.read_csv('batch_files\\new_claims.csv', sep=';', encoding='latin1')
    df = pd.read_csv('batch_files\\training_claims.csv', sep=';', encoding='latin1')

    #######################################################################################################################

    # We delete future variables to avoid endogeneity, and non-numeric variables
    delete_variables = ['hist_siniestro_actual_bbdd', 'hist_siniestro_actual_unidad_investigacion',
                        'hist_siniestro_actual_incidencia_tecnica', 'hist_siniestro_actual_incidencia_tecnica_positiva',
                        'hist_siniestro_actual_incidencias'] + STRING.Parameters.cp + STRING.Parameters.fecha

    for i in delete_variables:
        if i == 'id_siniestro':
            delete_variables.remove(i)
        if i in df:
            del df[i]

    # 11) APPEND AND MARK TEST
    df['TEST'] = pd.Series(0, index=df.index)
    df_test['TEST'] = pd.Series(1, index=df_test.index)
    df = pd.concat([df, df_test], axis=0, ignore_index=True)

    # We need to check if a new Variable for where we do not have information before exists
    # (It is impossible to train so we delete it)
    df_try = df[df['TEST'] == 0]
    len_col = len(df_try.index)
    delete_cols = []
    for i in df_try.columns.values.tolist():
        nan_values = df_try[i].isnull().sum()
        if nan_values == len_col:
            delete_cols.append(i)

    for i in delete_cols:
        del df[i]

    del df_try, delete_cols

    print('Nan Values with test ', df.isnull().sum().sum())

    # First, we fill the categorical variables because they are not NaN
    for i in STRING.fillna_vars:
        if i in df:
            df[i] = df[i].fillna(0)
    print('Nan Values after categoric fill ', df.isnull().sum().sum())

    # Second, we check a variable new that is a categorical process
    df_try = df[df['TEST'] == 1]
    len_col = len(df_try.index)
    for i in df_try.columns.values.tolist():
        nan_values = df_try[i].isnull().sum()
        if nan_values == len_col:
            df.loc[df['TEST'] == 1, i] = 0

    # Third, we fill the remaining with a mark of -1
    df = df.fillna(-1)

    # 6) ROBUST SCALE
    # First we separate TEST values
    df_base = df[df['TEST'] == 0]
    df_test = df[df['TEST'] == 1]

    # then compute the robust scale for TEST=0
    df_base, params = process_utils.robust_scale(df_base, quantile_range=(10.0, 90.0))

    # Finally we get the center and the scale from TEST=0 and apply to TEST=1
    columns_to_scale = params['column_name'].tolist()

    for i in columns_to_scale:
        center = params.loc[params['column_name'] == i, 'center'].iloc[0]
        scale = params.loc[params['column_name'] == i, 'scale'].iloc[0]
        df_test[i] = pd.to_numeric(df_test[i], errors='coerce')
        df_test[i] = df_test[i].fillna(-1)
        df_test[i] = (df_test[i] - center) / scale

    df = pd.concat([df_base, df_test], axis=0, ignore_index=True)

    # 7) PCA REDUCTION
    df = df.dropna(subset=['id_siniestro'])
    df = process_utils.pca_reduction(df, show_plot=True, variance=95.00)

    # 8) APPEND BLACKLIST (CLUSTER VARIABLE)
    df = process_utils.append_blacklist(df)

    # 9) CORRELATION
    # process_utils.correlation_get_column(df, columns=[], output_file=False, show_plot=False)
    # process_utils.correlation_get_all(df, get_all=False, get_specific='FRAUDE')

    # 10) NORMAL ANORMAL FILES
    normal, anomaly, new = process_utils.output_normal_anormal_new(df, output_file=True, input_name_file='raw_file')

    '-----------------------------------------TUNNING MODEL------------------------------------------------------------'

    # from models.gradient_boosting import gb
    # from models.RNC import RNC
    # from models.light_xgb import lgb
    # from models.stacking import StackingAveragedModels, StackTreshold

    normal = normal.drop_duplicates(subset='id_siniestro')
    anomaly = anomaly.drop_duplicates(subset='id_siniestro')
    new = new.drop_duplicates(subset='id_siniestro')

    # UNSUPERVISED MODEL SELECTION ####################################################################################
    # 1) First we calculate the Fraud Score for each model (you can try other commented models)
    oversample_times = round(len(normal.index) / len(anomaly.index))
    kmeans_fscore = uns.mini_batch_kmeans_tunning(normal, anomaly, oversample_times)
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
    kmeans_fscore = kmeans_fscore[kmeans_fscore['fraud_score'] == kmeans_fscore['fraud_score'].max()].reset_index(drop=True)

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
    new = new.drop_duplicates(subset='id_siniestro')
    new['id_siniestro'] = new['id_siniestro'].map(int)
    new = new.drop_duplicates(subset='id_siniestro')

    # Train - Test
    train, test = train_test_utils.train_test.training_test(normal, anomaly)
    train['id_siniestro'] = train['id_siniestro'].map(int)
    test['id_siniestro'] = test['id_siniestro'].map(int)
    train = pd.merge(train, comparative.drop(['FRAUDE'], axis=1), how='left', on='id_siniestro')
    test = pd.merge(test, comparative.drop(['FRAUDE'], axis=1), how='left', on='id_siniestro')
    train = train.drop_duplicates(subset='id_siniestro')
    test = test.drop_duplicates(subset='id_siniestro')

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
        del train[i]
        del test[i]
        del train_t[i]
        del test_t[i]
        del valid_t[i]

    for i in labels:
        if i in new.columns.values.tolist():
            del new[i]

    '''
    # 2) TUNNING PARAMETERS
    
    bootstrap, n_estimators, max_depth, oob_score, class_weight, sampling, min_sample_leaf, min_sample_split, max_features = \
        ert.extreme_randomize_tunning(train, test, sampling=None, label=selected_label)
    
    
    num_leaves, n_estimators_lgb, max_bin, sampling_lgb, min_sample_leaf_lgb = lgb.lgb_tunning(
        train, test, sampling='ADASYN', label=selected_label)
    
    
    n_estimators_gb, max_depth_gb, sampling_gb, min_sample_leaf_gb, min_sample_split_gb = gb.gb_tunning(train, test, sampling='ADASYN', label=selected_label)
    
    radius_rnc, leaf_size_rnc, sampling_rnc = RNC.rnc_tunning(train, test, sampling=None, label=selected_label)
    
    n_estimators_gb = 300
    max_depth_gb = 200
    sampling_gb = 'ADASYN'
    num_leaves = 2000
    n_estimators_lgb = 300
    max_bin = 500
    sampling_lgb = 'ADASYN'
    
    '''
    max_depth = 500
    oob_score = True
    class_weight = 'balanced_subsample'
    base_sampling = None
    control_sampling = 'ADASYN'
    bootstrap = True
    n_estimators = 500


    # 3) TUNNING THRESHOLD
    print('MODELO BASE THRESHOLD')
    final_tresh_base = ert.extreme_randomize_treshold(train_t.copy(), valid_t.copy(), test_t.copy(), comparative.copy(), bootstrap=bootstrap,
                                                 n_estimators=n_estimators,
                                                 max_depth=max_depth,
                                                 oob_score=oob_score,
                                                 class_weight=class_weight,
                                                 sampling=base_sampling,
                                                 label=selected_label, beta=2, create_folder=True)
    print('MODELO CONTROL THRESHOLD')
    final_tresh_control = ert.extreme_randomize_treshold(train_t.copy(), valid_t.copy(), test_t.copy(), comparative.copy(), bootstrap=bootstrap,
                                                 n_estimators=n_estimators,
                                                 max_depth=max_depth,
                                                 oob_score=oob_score,
                                                 class_weight=class_weight,
                                                 sampling=control_sampling,
                                                 label=selected_label, beta=2, create_folder=False)

    '''
    final_tresh_lightgb = lgb.lightgb_treshold(train_t, valid_t, test_t, comparative, num_leaves, n_estimators_lgb, max_bin, sampling_lgb,
                                               label=selected_label, beta=2)
    
    
    final_tresh_gb = gb.gb_treshold(train_t, valid_t, test_t, comparative, max_depth=max_depth_gb,
                                    n_estimators=n_estimators_gb, sampling=sampling_gb, learning_rate=0.005,
                                    label=selected_label, beta=2)
    
    
    final_tresh_rnc = RNC.rnc_treshold(train_t, valid_t, test_t, radius_rnc, leaf_size_rnc, sampling_rnc,
                                       label=selected_label, beta=2)
    
    
    
    # STACKING MODELS
    # Models
    min_sample_leaf = round((len(train_t.index)) * 0.005)
    min_sample_split = min_sample_leaf * 10
    max_features = round(len(train_t.columns)/3)
    yTrain = train_t[[selected_label]]
    xTrain = train_t
    
    ERT = ensemble.ExtraTreesClassifier(bootstrap=bootstrap, n_estimators=n_estimators, max_depth=max_depth,
                                        oob_score=oob_score, class_weight=class_weight, min_samples_leaf=min_sample_leaf,
                                        min_samples_split=min_sample_split, max_features='auto', n_jobs=-1)
    
    Gboost = ensemble.GradientBoostingClassifier(n_estimators=n_estimators_gb, learning_rate=0.005,
                                       max_depth=max_depth_gb, 
                                       loss='deviance', random_state=531, min_samples_split=min_sample_split,
                                                 min_samples_leaf=min_sample_leaf)
    
    # RNC = neighbors.RadiusNeighborsClassifier(radius=radius_rnc, weights='distance', algorithm='auto',
    #                                           leaf_size=leaf_size_rnc, p=2, outlier_label=-1)
    
    Light_Gboost = lgboost.LGBMClassifier(boosting_type="gbdt", num_leaves= num_leaves, max_depth= -1,
                                learning_rate= 0.005, n_estimators= n_estimators_lgb,
                                max_bin= max_bin,
                                objective= 'binary',
                                min_split_gain= 0., min_child_weight= 5, min_child_samples= min_sample_leaf,
                                subsample= 1., subsample_freq= 1, colsample_bytree= 1.,
                                reg_alpha= 0., reg_lambda= 0., random_state= 531,
                                n_jobs= -1, silent= False
                                )
    
    
    stacked_averaged_models1 = StackingClassifier(classifiers=[Gboost, Light_Gboost], meta_classifier=ERT)
    
    stacked_averaged_models2 = StackingClassifier(classifiers=[Gboost, ERT], meta_classifier=Light_Gboost)
    
    stacked_averaged_models3 = StackingClassifier(classifiers=[Light_Gboost, ERT], meta_classifier=Gboost)
    
    
    stacked_averaged_models.fit(xTrain.drop([selected_label] + ['id_siniestro'], axis=1).values, yTrain.values)
    y_pred_score = stacked_averaged_models.predict(valid_t.drop([selected_label] + ['id_siniestro'], axis=1).values)
    
    
    # Stacked Treshold
    stacked_treshold = StackTreshold.stacking_treshold(train_t, valid_t, test_t, stacked_averaged_models3, comparative,
                                                       label=selected_label, beta=2)
    
    
    
    
    # Stacked Evaluation
    min_sample_leaf = round((len(train.index)) * 0.005)
    min_sample_split = min_sample_leaf * 10
    max_features = round(len(train.columns)/3)
    ERT = ensemble.ExtraTreesClassifier(bootstrap=bootstrap, n_estimators=n_estimators, max_depth=max_depth,
                                        oob_score=oob_score, class_weight=class_weight, min_samples_leaf=min_sample_leaf,
                                        min_samples_split=min_sample_split, max_features='auto', n_jobs=-1)
    
    Gboost = ensemble.GradientBoostingClassifier(n_estimators=n_estimators_gb, learning_rate=0.005,
                                       max_depth=max_depth_gb, 
                                       loss='deviance', random_state=531, min_samples_split=min_sample_split,
                                                 min_samples_leaf=min_sample_leaf)
    
    Light_Gboost = lgboost.LGBMClassifier(boosting_type="gbdt", num_leaves= num_leaves, max_depth= -1,
                                learning_rate= 0.005, n_estimators= n_estimators_lgb,
                                max_bin= max_bin,
                                objective= 'binary',
                                min_split_gain= 0., min_child_weight= 5, min_child_samples= min_sample_leaf,
                                subsample= 1., subsample_freq= 1, colsample_bytree= 1.,
                                reg_alpha= 0., reg_lambda= 0., random_state= 531,
                                n_jobs= -1, silent= False
                                )
    
    
    StackTreshold.stacking_evaluation(train, test, comparative, stacked_treshold, stacked_averaged_models, label=selected_label, beta=2)
    
    # Stacked Appplied
    
    Train = pd.concat([train, test], axis=0, ignore_index=True)
    StackTreshold.stacking_applied(Train, new, po_res_indem, treshold=stacked_treshold, fileModel=stacked_averaged_models,
                                   n_random=10, label=selected_label)
    '''

    # 4) MODEL EVALUATION
    ert.extreme_randomize_evaluation(train.copy(), test.copy(), comparative.copy(), treshold=final_tresh_base, bootstrap=bootstrap,
                                     n_estimators=n_estimators,
                                     max_depth=max_depth,
                                     oob_score=oob_score,
                                     class_weight=class_weight,
                                     sampling=base_sampling,
                                     label=selected_label)


    '-----------------------------------------APPLICATION-------------------------------------------------------------'
    train = pd.concat([train, test], axis=0)
    test = new

    print('MODELO CONTROL APLICADO')
    ert.extreme_randomize_applied(train.copy(), test.copy(), treshold=final_tresh_control,
                                                         bootstrap=bootstrap, n_estimators=n_estimators, max_depth=max_depth,
                                                         oob_score=oob_score, class_weight=class_weight,
                                                         sampling=control_sampling, label=selected_label,
                                                        )

    print('MODELO BASE APLICADO')
    ert.extreme_randomize_applied(train, test, treshold=final_tresh_base,
                                                         bootstrap=bootstrap, n_estimators=n_estimators, max_depth=max_depth,
                                                         oob_score=oob_score, class_weight=class_weight,
                                                         sampling=base_sampling, label=selected_label,
                                                        )




if __name__ == '__main__':
    global_main()