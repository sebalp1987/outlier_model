import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold
from sklearn.metrics import fbeta_score, precision_score, recall_score, confusion_matrix
from utils.model_utils import plot_confusion_matrix
import pylab as plot
import pandas as pd


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=15)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    # Do the predictions of all base models on the test data and use the averaged predictions as
    # meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)
    
    
class StackTreshold:    
    def stacking_treshold(Train, Valid, Test, fileModel, comparative, label='FRAUDE', beta=2):

        # With beta = 2, we give the same importance to Recall and Precision
        import time
        t_0 = time.time()
        yTrain = Train[[label]]
        xTrain = Train
        del xTrain[label]

        names = Train.columns.values.tolist()
        fileNames = np.array(names)
        from utils.model_utils import over_sampling
        xTrain, yTrain = over_sampling(xTrain, yTrain, model='ADASYN')

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
            print(y_hat)
            y_hat = [item for sublist in y_hat for item in sublist]

            scores.append([
                recall_score(y_pred=y_hat, y_true=Valid[label].values),
                precision_score(y_pred=y_hat, y_true=Valid[label].values),
                fbeta_score(y_pred=y_hat, y_true=Valid[label].values,
                            beta=2)])

        scores = np.array(scores)
        print('max_scores', scores[:, 2].max(), scores[:, 2].argmax())
        t_1 = time.time()
        print(t_1 - t_0)
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
        plot_confusion_matrix(cnf_matrix, name='final_files\\confussion_FRAUDE_Clusters.png', classes=['No Fraude', 'Fraude'], title='Confusion matrix')

        cnf_matrix = confusion_matrix(Test['FRAUDE'].values, y_hat_test)
        plot_confusion_matrix(cnf_matrix, name='final_files\\confussion_FRAUDE.png', classes=['Normal', 'Anormal'], title='Confusion matrix')
        return final_tresh
    
    
    def stacking_evaluation(Train, Test, comparative, treshold, fileModel, label='FRAUDE', beta=2):

        yTrain = Train[label]
        xTrain = Train
        del xTrain[label]

        names = Train.columns.values.tolist()
        fileNames = np.array(names)
        from utils.model_utils import over_sampling
        xTrain, yTrain = over_sampling(xTrain, yTrain, model='ADASYN')

        fileModel.fit(xTrain.values, yTrain.values)
        y_hat_test = fileModel.predict_proba(Test.drop(label, axis=1).values)

        df_proba = pd.DataFrame(y_hat_test, index=Test.index)
        df_proba = pd.concat([Test, df_proba], axis=1)
        df_proba.columns = ['VALOR REAL', 'VALOR_PREDICHO_NO_FRAUDE', 'VALOR_PREDICHO_FRAUDE']
        df_proba.to_csv('final_files\\probabilidades_stacking.csv', sep=';', index=False, encoding='latin1')

        y_hat_test = np.delete(y_hat_test, 0, axis=1)

        y_hat_test = (y_hat_test > treshold).astype(int)
        y_hat_test = y_hat_test.tolist()
        y_hat_test = [item for sublist in y_hat_test for item in sublist]

        print('Final threshold: %.3f' % treshold)
        print('Test Recall Score: %.3f' % recall_score(y_pred=y_hat_test, y_true=Test[label].values))
        print('Test Precision Score: %.3f' % precision_score(y_pred=y_hat_test, y_true=Test[label].values))
        print('Test F2 Score: %.3f' % fbeta_score(y_pred=y_hat_test, y_true=Test[label].values, beta=beta))

        for i in comparative.columns.values.tolist():
            if i != 'id_siniestro' and i in Test.columns.values.tolist():
                del comparative[i]

        Test = pd.merge(Test, comparative, how='left', on='id_siniestro')
        cnf_matrix = confusion_matrix(Test['FRAUDE_Clusters'].values, y_hat_test)
        plot_confusion_matrix(cnf_matrix, classes=['No Fraude', 'Fraude'], title='Confusion matrix')

        cnf_matrix = confusion_matrix(Test['FRAUDE'].values, y_hat_test)
        plot_confusion_matrix(cnf_matrix, classes=['Normal', 'Anormal'], title='Confusion matrix')

        featureImportance = fileModel.feature_importances_

        featureImportance = featureImportance / featureImportance.max()

        sorted_idx = np.argsort(featureImportance)
        barPos = np.arange(sorted_idx.shape[0]) + 0.5
        plot.barh(barPos, featureImportance[sorted_idx], align='center')
        plot.yticks(barPos, fileNames[sorted_idx])
        plot.xlabel('Variable Importance')
        plot.show()

    def stacking_applied(Train, New, po_res_indem, treshold, fileModel, n_random=10, label='FRAUDE'):

        yTrain = Train[label]
        xTrain = Train
        del xTrain[label]

        names = Train.columns.values.tolist()
        fileNames = np.array(names)

        fileModel.fit(xTrain.values, yTrain.values)
        y_hat_New = fileModel.predict(New.drop('id_siniestro', axis=1).values)

        df_proba = pd.DataFrame(y_hat_New, index=New.index)
        df_proba = pd.concat([df_proba, New], axis=1)

        y_hat_New = np.delete(y_hat_New, 0, axis=1)

        y_random = (y_hat_New <= treshold).astype(int)
        y_hat_New = (y_hat_New > treshold).astype(int)
        df_proba_threshold = pd.DataFrame(y_hat_New, index=New.index)
        df_proba_conc = pd.concat([df_proba, df_proba_threshold], axis=1)
        df_proba_conc = pd.merge(df_proba_conc, po_res_indem, how='left', on='id_siniestro')
        df_proba_conc.to_csv('final_files\\probabilidades_stacking_new.csv', sep=';', index=False, encoding='latin1')

        # Randomly selection
        df_random = pd.DataFrame(y_random, index=New.index)
        df_random = df_random.sample(n=n_random)
        df_proba_conc = pd.concat([df_proba, df_random], axis=1)
        df_proba_conc = pd.merge(df_proba_conc, po_res_indem, how='left', on='id_siniestro')
        df_proba_conc.to_csv('final_files\\probabilidades_stacking_new2.csv', sep=';', index=False, encoding='latin1')


        featureImportance = fileModel.feature_importances_

        featureImportance = featureImportance / featureImportance.max()

        sorted_idx = np.argsort(featureImportance)
        barPos = np.arange(sorted_idx.shape[0]) + 0.5
        plot.barh(barPos, featureImportance[sorted_idx], align='center')
        plot.yticks(barPos, fileNames[sorted_idx])
        plot.xlabel('Variable Importance')
        plot.show()
