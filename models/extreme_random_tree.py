from sklearn import ensemble
import pylab as plot
import pandas as pd
from utils.model_utils import plot_confusion_matrix
from sklearn.metrics import fbeta_score, precision_score, recall_score, confusion_matrix
import numpy as np
from utils.model_utils import under_sampling, over_sampling
from sklearn.model_selection import GridSearchCV



_author_ = 'Sebastian Palacio'


class extreme_randomize:

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
        min_sample_split = min_sample_leaf*10

        features = round(len(xTrain.columns)/3)

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

        return max_depth, n_estimators, bootstrap, oob_score, class_weight, max_features, sampling, min_sample_leaf, min_sample_split, max_features

    def extreme_randomize_treshold(Train, Valid, Test, comparative, bootstrap=False, n_estimators: int=200, max_depth: int=50
                              , oob_score: bool=False, class_weight='balanced_subsample',
                                   sampling=None, label='FRAUDE', beta=2):

        # With beta = 2, we give the same importance to Recall and Precision
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

        model_name = str(sampling)

        min_sample_leaf = round((len(xTrain.index)) * 0.005)
        min_sample_split = min_sample_leaf * 10
        max_features = round(len(xTrain.columns)/3)

        fileModel = ensemble.ExtraTreesClassifier(criterion='entropy', bootstrap=bootstrap,
                                                  min_samples_leaf=min_sample_leaf,
                                                  min_samples_split=min_sample_split,
                                                  n_estimators=n_estimators,
                                                  max_depth=max_depth, max_features=max_features,
                                                  oob_score=oob_score,
                                                  random_state=531, verbose=1, class_weight=class_weight,
                                                  n_jobs=1)

        fileModel.fit(xTrain.drop(['id_siniestro'], axis=1).values, yTrain.values)

        print(np.median(fileModel.predict_proba(Valid[Valid[label] == 0].drop([label] + ['id_siniestro'], axis=1).values)))
        print(np.median(fileModel.predict_proba(Valid[Valid[label] == 1].drop([label] + ['id_siniestro'], axis=1).values)))

        tresholds = np.linspace(0.1, 1.0, 200)

        scores = []

        y_pred_score = fileModel.predict_proba(Valid.drop([label] + ['id_siniestro'], axis=1).values)
        print(fileModel.classes_)
        print(y_pred_score)
        y_pred_score = np.delete(y_pred_score, 0, axis=1)

        print('min', y_pred_score.min())
        print('max', y_pred_score.max())

        for treshold in tresholds:
            y_hat = (y_pred_score > treshold).astype(int)
            y_hat = y_hat.tolist()
            y_hat = [item for sublist in y_hat for item in sublist]


            scores.append([
                recall_score(y_pred=y_hat, y_true=Valid[label].values),
                precision_score(y_pred=y_hat, y_true=Valid[label].values),
                fbeta_score(y_pred=y_hat, y_true=Valid[label].values,
                            beta=2)])

        scores = np.array(scores)
        print('max_scores', scores[:, 2].max(), scores[:, 2].argmax())

        plot.plot(tresholds, scores[:, 0], label='$Recall$')
        plot.plot(tresholds, scores[:, 1], label='$Precision$')
        plot.plot(tresholds, scores[:, 2], label='$F_2$')
        plot.ylabel('Score')
        plot.xlabel('Threshold')
        plot.legend(loc='best')
        file_name = 'ERT_treshold_' + str(sampling) + '.png'
        plot.close()

        final_tresh = tresholds[scores[:, 2].argmax()]

        y_hat_test = fileModel.predict_proba(Test.drop([label] + ['id_siniestro'], axis=1).values)
        y_hat_test = np.delete(y_hat_test, 0, axis=1)

        y_hat_test = (y_hat_test > final_tresh).astype(int)
        y_hat_test = y_hat_test.tolist()
        y_hat_test = [item for sublist in y_hat_test for item in sublist]

        print('Final threshold: %.3f' % final_tresh)
        print('Test Recall Score: %.3f' % recall_score(y_pred=y_hat_test, y_true=Test[label].values))
        print('Test Precision Score: %.3f' % precision_score(y_pred=y_hat_test, y_true=Test[label].values))
        print('Test F2 Score: %.3f' % fbeta_score(y_pred=y_hat_test, y_true=Test[label].values, beta=beta))

        del comparative['FRAUDE_Clusters']
        Test = pd.merge(Test, comparative, how='left', on='id_siniestro')
        cnf_matrix = confusion_matrix(Test['FRAUDE_Clusters'].values, y_hat_test)
        plot_confusion_matrix(cnf_matrix, classes=['No Fraude', 'Fraude'],
                              title='Confusion matrix')

        cnf_matrix = confusion_matrix(Test['FRAUDE'].values, y_hat_test)
        plot_confusion_matrix(cnf_matrix, classes=['Normal', 'Anormal'],
                              title='Confusion matrix')

        return final_tresh

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

        fileModel.fit(xTrain.drop(['id_siniestro'], axis=1).values, yTrain.values)
        y_hat_test = fileModel.predict_proba(Test.drop([label] + ['id_siniestro'], axis=1).values)

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

    def extreme_randomize_applied(Train, New, treshold, bootstrap=False, n_estimators: int = 200, max_depth: int = 50,
                                  oob_score: bool = False, class_weight='balanced_subsample',
                                  sampling=None, label='FRAUDE'):

        yTrain = Train[[label]]
        xTrain = Train
        del xTrain[label]

        if sampling == None:
            pass
        elif sampling == 'ALLKNN':
            xTrain, yTrain = under_sampling(xTrain, yTrain)
        else:
            xTrain, yTrain = over_sampling(xTrain, yTrain, model=sampling)

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

        fileModel.fit(xTrain.drop(['id_siniestro'], axis=1).values, yTrain.values)
        y_hat_New = fileModel.predict_proba(New.drop('id_siniestro', axis=1).values)
        y_hat_New = np.delete(y_hat_New, 0, axis=1)
        df_proba = pd.DataFrame(y_hat_New, columns=['probabilidad'], index=New.index)
        df_proba = pd.concat([New['id_siniestro'], df_proba], axis=1)

        # y_random = (y_hat_New <= treshold).astype(int)
        y_hat_New = (y_hat_New > treshold).astype(int)
        print(y_hat_New)
        df_proba_threshold = pd.DataFrame(y_hat_New, columns=['Treshold'], index=New.index)
        print(df_proba_threshold)
        df_proba_conc = pd.concat([df_proba, df_proba_threshold], axis=1)
        print(df_proba_conc)
        return df_proba_conc

