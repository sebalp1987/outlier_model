from sklearn.cross_validation import train_test_split
import lightgbm as lgbx
from sklearn.metrics import mean_squared_error, r2_score
import pylab as plot
import numpy
import pandas as pd
from utils.model_utils import plot_confusion_matrix
from sklearn.metrics import fbeta_score, precision_score, recall_score, confusion_matrix
import numpy as np
from utils.model_utils import under_sampling, over_sampling
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict


class lgb:

    def lgb_tunning(Train, Test,  sampling=None, scores='f1', label='FRAUDE'):

        Train = pd.concat([Train, Test], axis=0, ignore_index=True)
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

        tuned_parameters = [{'boosting_type': ["gbdt"], 'num_leaves': list(numpy.arange(100, 1000, 100)), 'max_depth':[-1],
                            'learning_rate': [0.005], 'n_estimators': list(numpy.arange(200, 500, 100)),
                            'max_bin': list(numpy.arange(200, 500, 100)),
                            'objective': ['binary'],
                            'min_split_gain': [0.], 'min_child_weight': [5], 'min_child_samples': [min_sample_leaf],
                            'subsample': [1.], 'subsample_freq': [1], 'colsample_bytree': [1.],
                            'reg_alpha': [0.], 'reg_lambda': [0.], 'random_state': [531],
                            'n_jobs': [-1], 'silent': [False]
                            }]

        fileModel = GridSearchCV(lgbx.LGBMClassifier(), param_grid=tuned_parameters, cv=10,
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

        num_leaves = int(dict_values['num_leaves'])
        n_estimators = int(dict_values['n_estimators'])
        max_bin = int(dict_values['max_bin'])

        df = pd.DataFrame.from_dict(dict_values, orient="index")
        df.to_csv('final_files\\sup_lgb.csv', sep=';', encoding='latin1', index=False)

        return num_leaves, n_estimators, max_bin, sampling, min_sample_leaf


    def lightgb_treshold(Train, Valid, Test, comparative, num_leaves, n_estimators, max_bin,
                                   sampling=None, label='FRAUDE', beta=2):

        # With beta = 2, we give the same importance to Recall and Precision


        yTrain = Train[[label]]
        xTrain = Train
        del xTrain[label]

        names = Train.columns.values.tolist()
        fileNames = numpy.array(names)

        if sampling == None:
            pass
        elif sampling == 'ALLKNN':
            xTrain, yTrain = under_sampling(xTrain, yTrain)
        else:
            xTrain, yTrain = over_sampling(xTrain, yTrain, model=sampling)

        min_sample_leaf = round((len(xTrain.index)) * 0.005)

        fileModel = lgbx.LGBMClassifier(boosting_type= "gbdt", num_leaves= num_leaves, max_depth=-1,
                            learning_rate= 0.005, n_estimators= n_estimators,
                            max_bin= max_bin,
                            objective= 'binary',
                            min_split_gain= 0., min_child_weight= 5, min_child_samples= min_sample_leaf,
                            subsample= 1., subsample_freq= 1, colsample_bytree= 1.,
                            reg_alpha= 0., reg_lambda= 0., random_state= 531,
                            n_jobs= -1, silent= False)

        fileModel.fit(xTrain.drop(['id_siniestro'], axis=1).values, yTrain.values)

        print(np.median(fileModel.predict_proba(Valid[Valid[label] == 0].drop([label] + ['id_siniestro'], axis=1).values)))
        print(np.median(fileModel.predict_proba(Valid[Valid[label] == 1].drop([label] + ['id_siniestro'], axis=1).values)))

        tresholds = np.linspace(0.1, 1.0, 200)

        scores = []

        y_pred_score = fileModel.predict_proba(Valid.drop([label] + ['id_siniestro'], axis=1).values)
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
        plot.show()

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

        featureImportance = fileModel.feature_importances_

        featureImportance = featureImportance / featureImportance.max()

        sorted_idx = numpy.argsort(featureImportance)
        barPos = numpy.arange(sorted_idx.shape[0]) + 0.5
        plot.barh(barPos, featureImportance[sorted_idx], align='center')
        plot.yticks(barPos, fileNames[sorted_idx])
        plot.xlabel('Variable Importance')
        plot.show()

        return final_tresh
