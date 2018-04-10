
from sklearn import neighbors
import pylab as plot
import numpy
import pandas as pd
from utils.model_utils import plot_confusion_matrix
from sklearn.metrics import fbeta_score, precision_score, recall_score, confusion_matrix
import numpy as np
from utils.model_utils import under_sampling, over_sampling
from sklearn.model_selection import GridSearchCV



class RNC:

    def rnc_tunning(Train, Test,  sampling=None, scores='f1', label='FRAUDE'):

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

        tuned_parameters = [{'radius': list(numpy.arange(1, 101, 1)),
                             'weights': ['distance'], 'algorithm': ['auto'],
                             'leaf_size': list(numpy.arange(30, 301, 30)),
                             'p': [2],
                             'outlier_label': [-1]
                             }]

        fileModel = GridSearchCV(neighbors.RadiusNeighborsClassifier(), param_grid=tuned_parameters, cv=10,
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

        radius = int(dict_values['radius'])
        leaf_size = int(dict_values['leaf_size'])

        df = pd.DataFrame.from_dict(dict_values, orient="index")
        df.to_csv('final_files\\sup_rnc.csv', sep=';', encoding='latin1', index=False)

        return radius, leaf_size, sampling
    
    
    def rnc_treshold(Train, Valid, Test, radius, leaf_size,
                                   sampling=None, label='FRAUDE', beta=2):

        # With beta = 2, we give the same importance to Recall and Precision

        yTrain = Train[label]
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

        fileModel = neighbors.RadiusNeighborsClassifier(radius= radius,
                             weights= 'distance', algorithm= 'auto',
                             leaf_size= leaf_size,
                             p= 2,
                             outlier_label= -1)

        fileModel.fit(xTrain.values, yTrain.values)

        print(np.median(fileModel.predict_proba(Valid[Valid[label] == 0].drop(label, axis=1).values)))
        print(np.median(fileModel.predict_proba(Valid[Valid[label] == 1].drop(label, axis=1).values)))

        tresholds = np.linspace(0.1, 1.0, 200)

        scores = []

        y_pred_score = fileModel.predict_proba(Valid.drop(label, axis=1).values)
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

        y_hat_test = fileModel.predict_proba(Test.drop(label, axis=1).values)
        y_hat_test = np.delete(y_hat_test, 0, axis=1)

        y_hat_test = (y_hat_test > final_tresh).astype(int)
        y_hat_test = y_hat_test.tolist()
        y_hat_test = [item for sublist in y_hat_test for item in sublist]

        print('Final threshold: %.3f' % final_tresh)
        print('Test Recall Score: %.3f' % recall_score(y_pred=y_hat_test, y_true=Test[label].values))
        print('Test Precision Score: %.3f' % precision_score(y_pred=y_hat_test, y_true=Test[label].values))
        print('Test F2 Score: %.3f' % fbeta_score(y_pred=y_hat_test, y_true=Test[label].values, beta=beta))

        cnf_matrix = confusion_matrix(Test[label].values, y_hat_test)
        plot_confusion_matrix(cnf_matrix, classes=['Normal', 'Anormal'], title='Confusion matrix')

        featureImportance = fileModel.feature_importances_

        featureImportance = featureImportance / featureImportance.max()

        sorted_idx = numpy.argsort(featureImportance)
        barPos = numpy.arange(sorted_idx.shape[0]) + 0.5
        plot.barh(barPos, featureImportance[sorted_idx], align='center')
        plot.yticks(barPos, fileNames[sorted_idx])
        plot.xlabel('Variable Importance')
        plot.show()

        return final_tresh

