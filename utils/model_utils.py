from matplotlib import pyplot as plot
import itertools
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.under_sampling import AllKNN
import pandas as pd
import numpy as np

def plot_confusion_matrix(cm, classes,  title='Confusion matrix', cmap=plot.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Copyed from a kernel by joparga3 https://www.kaggle.com/joparga3/kernels
    """
    plot.figure()
    plot.imshow(cm, interpolation='nearest', cmap=cmap)
    plot.title(title)
    plot.colorbar()
    tick_marks = np.arange(len(classes))
    plot.xticks(tick_marks, classes, rotation=0)
    plot.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plot.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plot.tight_layout()
    plot.ylabel('True label')
    plot.xlabel('Predicted label')
    plot.show()



def over_sampling(xTrain, yTrain, model='ADASYN', neighbors=200):
    """
    It generate synthetic sampling for the minority class using the model specificed. Always it has
    to be applied to the training set.
    :param xTrain: X training set.
    :param yTrain: Y training set.
    :param model: 'ADASYN' or 'SMOTE'
    :param neighbors: number of nearest neighbours to used to construct synthetic samples.
    :return: xTrain and yTrain oversampled
    """

    if model == 'ADASYN':
        model = ADASYN(random_state=42, ratio='minority', n_neighbors=neighbors)

    if model == 'SMOTE':
        model = SMOTE(random_state=42, ratio='minority', k_neighbors=neighbors, m_neighbors='svm')

    xTrain, yTrain = model.fit_sample(xTrain, yTrain)

    return xTrain, yTrain


def under_sampling(xTrain, yTrain, neighbors=200):
    """
    It reduces the sample size for the majority class using the model specificed. Always it has
    to be applied to the training set.
    :param xTrain: X training set.
    :param yTrain: Y training set.
    :param neighbors: size of the neighbourhood to consider to compute the
        average distance to the minority point samples
    :return: xTrain and yTrain oversampled
    """

    xTrainNames = xTrain.columns.values.tolist()
    yTrainNames = yTrain.columns.values.tolist()

    model = AllKNN(random_state=42, ratio='majority', n_neighbors=neighbors)

    xTrain, yTrain = model.fit_sample(xTrain, yTrain)

    xTrain = pd.DataFrame(xTrain, columns=[xTrainNames])
    yTrain = pd.DataFrame(yTrain, columns=[yTrainNames])

    return xTrain, yTrain


def oversample_unsupervised(normal, anomaly):
    X = pd.concat([normal, anomaly], axis=0)
    X = X.copy()
    X = X.reset_index(drop=True)
    Y_fraude = X[['FRAUDE']]
    id_claims = X[['id_siniestro']]
    del X['id_siniestro']
    del X['FRAUDE']
    xTrain_name = X.columns.values.tolist()
    xTrain, yTrain = over_sampling(X, Y_fraude, model='SMOTE')
    print(xTrain)
    print(yTrain)
    xTrain = pd.DataFrame(xTrain, columns=xTrain_name)
    yTrain = pd.DataFrame(yTrain, columns=['FRAUDE'])
    yTrain = pd.concat([yTrain, xTrain], axis=1)
    print(yTrain)
    yTrain = pd.concat([yTrain, id_claims], axis=1)
    yTrain['id_siniestro'] = yTrain['id_siniestro'].fillna(-1)
    normal = yTrain[yTrain['FRAUDE'] == 0]
    anomaly = yTrain[yTrain['FRAUDE'] == 1]

    return normal, anomaly