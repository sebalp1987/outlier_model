from sklearn import ensemble
import pylab as plot
import pandas as pd
from utils.model_utils import plot_confusion_matrix
from sklearn.metrics import fbeta_score, precision_score, recall_score, confusion_matrix
import numpy as np
from utils.model_utils import under_sampling, over_sampling
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgbx
from mlxtend.classifier import StackingClassifier

_author_ = 'Sebastian Palacio'


class Models:

    def extreme_randomize_tunning(Train, Test, sampling=None, scores='f1', label='FRAUDE'):

        Train = pd.concat([Train, Test], axis=0, ignore_index=True)
        yTrain = Train[[label]]
        xTrain = Train

        del xTrain[label]

        if sampling == None:
            pass
        elif sampling == 'ALLKNN':
            xTrain, yTrain = under_sampling(xTrain, yTrain)
            class_weight = None
        else:
            xTrain, yTrain = over_sampling(xTrain, yTrain, model=sampling)
            class_weight = None

        min_sample_leaf = round((len(xTrain.index)) * 0.005)
        min_sample_split = min_sample_leaf * 10

        features = round(len(xTrain.columns) / 3)

        if sampling != None:
            tuned_parameters = [{'bootstrap': [True, False], 'min_samples_leaf': [min_sample_leaf],
                                 'min_samples_split': [min_sample_split], 'n_estimators': [200, 300],
                                 'max_depth': [50, 100],
                                 'max_features': [features],
                                 'oob_score': [True, False], 'random_state': [531], 'verbose': [1]
                                 }]

        if sampling == None:
            tuned_parameters = [{'bootstrap': [True], 'min_samples_leaf': [min_sample_leaf],
                                 'min_samples_split': [min_sample_split], 'n_estimators': [200, 300],
                                 'max_depth': [50, 100],
                                 'max_features': [features],
                                 'oob_score': [True, False], 'random_state': [531], 'verbose': [1],
                                 'class_weight': ['balanced',
                                                  'balanced_subsample',
                                                  {0: 1, 1: 2}, {0: 1, 1: 3}, {0: 1, 1: 5}, {0: 1, 1: 10},
                                                  {0: 1, 1: 20},
                                                  {0: 1, 1: 30}, {0: 1, 1: 40}, {0: 1, 1: 50}, {0: 1, 1: 60},
                                                  {0: 1, 1: 70},
                                                  {0: 1, 1: 80}, {0: 1, 1: 90}, {0: 1, 1: 100}, {0: 1, 1: 110},
                                                  {0: 1, 1: 120},
                                                  {0: 1, 1: 130}, {0: 1, 1: 140}, {0: 1, 1: 142}
                                                  ],
                                 'n_jobs': [-1]
                                 }]

        fileModel = GridSearchCV(ensemble.ExtraTreesClassifier(), param_grid=tuned_parameters, cv=10,
                                 scoring='%s_macro' % scores)

        fileModel.fit(xTrain.drop(['id_siniestro'], axis=1).values, yTrain[label].values)

        print("Best parameters set found on development set:")
        print()
        dict_values = fileModel.best_params_
        print(dict_values)
        print()
        print("Grid scores on development set:")
        print()
        means = fileModel.cv_results_['mean_test_score']
        stds = fileModel.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, fileModel.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))

        max_depth = int(dict_values['max_depth'])
        n_estimators = int(dict_values['n_estimators'])
        bootstrap = bool(dict_values['bootstrap'])
        oob_score = bool(dict_values['oob_score'])
        class_weight = dict_values['class_weight']
        max_features = int(dict_values['max_features'])

        df = pd.DataFrame.from_dict(dict_values, orient="index")
        df.to_csv('final_files\\sup_ert.csv', sep=';', encoding='latin1', index=False)

        return max_depth, n_estimators, bootstrap, oob_score, class_weight, max_features, sampling, min_sample_leaf, \
               min_sample_split, max_features

    def model_evaluation(Train, Valid, Test, comparative, bootstrap=False, n_estimators: int = 200,
                                   max_depth: int = 50
                                   , oob_score: bool = False, class_weight='balanced_subsample',
                                   sampling=None, label='FRAUDE', model='ert'):

        # With beta = 2, we give the same importance to Recall and Precision
        if sampling is not None:
            class_weight = None
        model_name = str(sampling)

        # fileModel.fit(xTrain.drop(['id_siniestro'], axis=1).values, yTrain.values)

        # print(np.median(fileModel.predict_proba(Valid[Valid[label] == 0].drop([label] + ['id_siniestro'], axis=1).values)))
        # print(np.median(fileModel.predict_proba(Valid[Valid[label] == 1].drop([label] + ['id_siniestro'], axis=1).values)))

        tresholds = np.linspace(0.1, 1.0, 200)

        scores = []
        y_pred_score = np.empty(shape=[0, 2])
        predicted_index = np.empty(shape=[0, ])
        # y_pred_score = fileModel.predict_proba(Valid.drop([label] + ['id_siniestro'], axis=1).values)
        skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=False)
        Test = pd.concat([Train, Valid, Test], axis=0).reset_index()
        print(Test.shape)
        X = Test.drop([label] + ['id_siniestro'], axis=1)
        y = Test[[label]]
        for train_index, test_index in skf.split(X.values, y[label].values):
            X_train, X_test = X.loc[train_index].values, X.loc[test_index].values
            y_train, y_test = y.loc[train_index].values, y.loc[test_index].values
            if sampling == None:
                pass
            elif sampling == 'ALLKNN':
                X_train, y_train = under_sampling(X_train, y_train)
                class_weight = None
            else:
                X_train, y_train = over_sampling(X_train, y_train, model=sampling)
                class_weight = None

            min_sample_leaf = round(y_train.shape[0] * 0.005)
            min_sample_split = min_sample_leaf * 10
            max_features = round(X_train.shape[1] / 3)
            if model == 'ert':
                fileModel = ensemble.ExtraTreesClassifier(criterion='entropy', bootstrap=bootstrap,
                                                          min_samples_leaf=min_sample_leaf,
                                                          min_samples_split=min_sample_split,
                                                          n_estimators=n_estimators,
                                                          max_depth=max_depth, max_features=max_features,
                                                          oob_score=oob_score,
                                                          random_state=531, verbose=1, class_weight=class_weight,
                                                          n_jobs=-1)
            elif model == 'gb':
                fileModel = ensemble.GradientBoostingClassifier(loss='deviance', learning_rate=0.01,
                                                                n_estimators=200,
                                                                subsample=1.0, criterion='friedman_mse',
                                                                min_samples_split=min_sample_split,
                                                                min_samples_leaf=min_sample_leaf,
                                                                min_weight_fraction_leaf=0.,
                                                                max_depth=max_depth, min_impurity_decrease=0.,
                                                                min_impurity_split=None, init=None,
                                                                random_state=531, max_features=None, verbose=1,
                                                                max_leaf_nodes=None, warm_start=False,
                                                                presort='auto')
            elif model == 'lxgb':
                fileModel = lgbx.LGBMClassifier(boosting_type="gbdt", num_leaves=2000, max_depth=200,
                                                learning_rate=0.005, n_estimators=300,
                                                max_bin=500,
                                                objective='binary',
                                                min_split_gain=0., min_child_weight=5,
                                                min_child_samples=min_sample_leaf,
                                                subsample=1., subsample_freq=1, colsample_bytree=1.,
                                                reg_alpha=0., reg_lambda=0., random_state=531,
                                                n_jobs=-1, silent=True)

            elif model.startswith('stacked'):

                ERT = ensemble.ExtraTreesClassifier(bootstrap=bootstrap, n_estimators=n_estimators, max_depth=max_depth,
                                                    oob_score=oob_score, class_weight=class_weight,
                                                    min_samples_leaf=min_sample_leaf,
                                                    min_samples_split=min_sample_split, max_features='auto', n_jobs=-1)

                Gboost = ensemble.GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=0.005,
                                                             max_depth=max_depth,
                                                             loss='deviance', random_state=531,
                                                             min_samples_split=min_sample_split,
                                                             min_samples_leaf=min_sample_leaf)

                Light_Gboost = lgbx.LGBMClassifier(boosting_type="gbdt", num_leaves=2000, max_depth=-1,
                                                   learning_rate=0.005, n_estimators=300,
                                                   max_bin=500,
                                                   objective='binary',
                                                   min_split_gain=0., min_child_weight=5,
                                                   min_child_samples=min_sample_leaf,
                                                   subsample=1., subsample_freq=1, colsample_bytree=1.,
                                                   reg_alpha=0., reg_lambda=0., random_state=531,
                                                   n_jobs=-1, silent=False
                                                   )
                if model.endswith('_ERT'):
                    fileModel = StackingClassifier(classifiers=[Gboost, Light_Gboost],
                                                   meta_classifier=ERT, average_probas=True,
                                                   use_probas=True)
                elif model.endswith('_GB'):
                    fileModel = StackingClassifier(classifiers=[ERT, Light_Gboost],
                                                   meta_classifier=Gboost, average_probas=True, use_probas=True)
                elif model.endswith('_LXGB'):
                    fileModel = StackingClassifier(classifiers=[ERT, Gboost],
                                                   meta_classifier=Light_Gboost, average_probas=True, use_probas=True)

            fileModel.fit(X_train, y_train)
            y_pred_score_i = fileModel.predict_proba(X_test)
            y_pred_score = np.append(y_pred_score, y_pred_score_i, axis=0)
            print(y_pred_score.shape)
            print(test_index.shape)
            print(predicted_index.shape)
            predicted_index = np.append(predicted_index, test_index, axis=0)
            print(predicted_index)
            del X_train, X_test, y_train, y_test

        y_pred_score = np.delete(y_pred_score, 0, axis=1)
        print('min', y_pred_score.min())
        print('max', y_pred_score.max())

        for treshold in tresholds:
            y_hat = (y_pred_score > treshold).astype(int)
            y_hat = y_hat.tolist()
            y_hat = [item for sublist in y_hat for item in sublist]

            scores.append([
                recall_score(y_pred=y_hat, y_true=Test[label].values),
                precision_score(y_pred=y_hat, y_true=Test[label].values),
                fbeta_score(y_pred=y_hat, y_true=Test[label].values,
                            beta=2)])

        scores = np.array(scores)
        print('F-Score', scores[:, 2].max(), scores[:, 2].argmax())
        print('scores', scores[scores[: 2].argmax()])
        print(scores)

        plot.plot(tresholds, scores[:, 0], label='$Recall$')
        plot.plot(tresholds, scores[:, 1], label='$Precision$')
        plot.plot(tresholds, scores[:, 2], label='$F_2$')
        plot.ylabel('Score')
        plot.xlabel('Threshold')
        plot.legend(loc='best')
        plot.show()
        plot.close()

        final_tresh = tresholds[scores[:, 2].argmax()]
        print(final_tresh)

        y_hat_test = (y_pred_score > final_tresh).astype(int)
        y_hat_test = y_hat_test.tolist()
        y_hat_test = [item for sublist in y_hat_test for item in sublist]

        Test['id_siniestro'] = Test['id_siniestro'].map(int)
        comparative['id_siniestro'] = comparative['id_siniestro'].map(int)
        Test = pd.merge(Test, comparative[['id_siniestro', 'FRAUDE']], how='left', on='id_siniestro')
        cnf_matrix = confusion_matrix(Test['FRAUDE_Clusters'].values, y_hat_test)
        plot_confusion_matrix(cnf_matrix, classes=['Normal', 'Abnormal'],
                              title='Confusion matrix')

        cnf_matrix = confusion_matrix(Test['FRAUDE'].values, y_hat_test)
        plot_confusion_matrix(cnf_matrix, classes=['Unknown', 'Fraud'],
                              title='Confusion matrix')

        return None

    def extreme_randomize_evaluation(Train, Test, comparative, treshold, bootstrap=False, n_estimators: int = 200,
                                     max_depth: int = 50,
                                     oob_score: bool = False,
                                     class_weight='balanced_subsample',
                                     sampling=None, label='FRAUDE', beta=2):

        yTrain = Train[[label]]
        xTrain = Train
        del xTrain[label]

        if sampling == None:
            pass
        elif sampling == 'ALLKNN':
            class_weight = None
            xTrain, yTrain = under_sampling(xTrain, yTrain)
        else:
            xTrain, yTrain = over_sampling(xTrain, yTrain, model=sampling)
            class_weight = None

        model_name = str(sampling)

        min_sample_leaf = round((len(xTrain.index)) * 0.005)
        min_sample_split = min_sample_leaf * 10
        max_features = round(len(xTrain.columns) / 3)

        fileModel = ensemble.ExtraTreesClassifier(criterion='entropy', bootstrap=bootstrap,
                                                  min_samples_leaf=min_sample_leaf,
                                                  min_samples_split=min_sample_split,
                                                  n_estimators=n_estimators,
                                                  max_depth=max_depth, max_features=max_features,
                                                  oob_score=oob_score,
                                                  random_state=531, verbose=1, class_weight=class_weight,
                                                  n_jobs=1)

        # fileModel.fit(xTrain.drop(['id_siniestro'], axis=1).values, yTrain.values)
        # y_hat_test = fileModel.predict_proba(Test.drop([label] + ['id_siniestro'], axis=1).values)
        cv = StratifiedKFold(n_splits=5, random_state=42)

        y_hat_test = cross_val_predict(fileModel, Test.drop([label] + ['id_siniestro'], axis=1).values,
                                       Test[[label]].values, cv=cv, method='predict_proba')
        y_hat_test = np.delete(y_hat_test, 0, axis=1)

        y_hat_test = (y_hat_test > treshold).astype(int)
        y_hat_test = y_hat_test.tolist()
        y_hat_test = [item for sublist in y_hat_test for item in sublist]

        print('Final threshold: %.3f' % treshold)
        print('Test Recall Score: %.3f' % recall_score(y_pred=y_hat_test, y_true=Test[label].values))
        print('Test Precision Score: %.3f' % precision_score(y_pred=y_hat_test, y_true=Test[label].values))
        print('Test F2 Score: %.3f' % fbeta_score(y_pred=y_hat_test, y_true=Test[label].values, beta=beta))

        del comparative['FRAUDE_Clusters']
        Test = pd.merge(Test, comparative, how='left', on='id_siniestro')
        cnf_matrix = confusion_matrix(Test['FRAUDE_Clusters'].values, y_hat_test)

        plot_confusion_matrix(cnf_matrix,
                              classes=['No Fraude', 'Fraude'], title='Confusion matrix')

        cnf_matrix = confusion_matrix(Test['FRAUDE'].values, y_hat_test)
        plot_confusion_matrix(cnf_matrix, classes=['Normal', 'Anormal'],
                              title='Confusion matrix')

