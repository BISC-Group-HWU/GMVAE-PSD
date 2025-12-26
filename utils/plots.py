"""
---------------------------------------------------------------------
-- Author: Jhosimar George Arias Figueroa
---------------------------------------------------------------------

Util functions for partitioning input data

"""

import numpy as np
from skimage.transform import resize
import tkinter
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
from sklearn.metrics import confusion_matrix
import time
# from tensorflow.python.ops.numpy_ops import np_config
# np_config.enable_numpy_behavior()

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from collections import Counter


def im_psnr(img1, img2, max_value=1):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float32) -
                   np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))


def plot_reconstruction(recon_test, z_test, y_test, num_digits, name):
    h, w = 64, 64
    n = 20
    count = 0
    ave_psnr = [0] * n * num_digits
    canvas_x = np.empty((h*num_digits*2, w*n))
    for i in range(num_digits):
        gt = z_test[y_test == i]
        pred = recon_test[y_test == i]
        for j in range(n):
            canvas_x[2*h*i:2*h*i+h, j*w:(j+1)*w] = gt[j].reshape((64, 64))
            canvas_x[2*h*i+h:2*h*i+h+h, j *
                     w:(j+1)*w] = pred[j].reshape((64, 64))
            ave_psnr[count] = im_psnr(gt[j], pred[j])
            count += 1
    print("Average PSNR: %.5lf" % (np.mean(ave_psnr)))
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(canvas_x)
    # plt.colorbar()
    plt.close(fig)
    # plt.show()
    mean_psnr = np.mean(ave_psnr)
    fig.savefig(name+str(round(mean_psnr,2)).zfill(4)+'.png')
    return mean_psnr

def plot_reconstruction_realdata(images,recon_test, z_test, name):
    num_digits = 1
    h, w = 64, 64
    n = 20
    count = 0
    ave_psnr = [0] * n * num_digits
    canvas_x = np.empty((h*num_digits*3, w*n))
    for i in range(1):
        gt = z_test
        pred = recon_test
        for j in range(n):
            canvas_x[2*h*i:2*h*i+h,        j*w:(j+1)*w] = gt[j].reshape((64, 64))
            canvas_x[2*h*i+h:2*h*i+h+h,    j*w:(j+1)*w] = images[j].reshape((64, 64))
            canvas_x[2*h*i+2*h:2*h*i+h+2*h,j*w:(j+1)*w] = pred[j].reshape((64, 64))
            ave_psnr[count] = im_psnr(gt[j], pred[j])
            count += 1
    print("Average PSNR: %.5lf" % (np.mean(ave_psnr)))
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(canvas_x)
    # plt.colorbar()
    plt.close(fig)
    # plt.show()
    ave_snr = np.mean(ave_psnr)
    fig.savefig(name+str(round(ave_snr,2))+'.png')
    return np.mean(ave_psnr)
    

def plot_latentspace(embedding, y, name):
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=y, marker='o',
                edgecolor='none', cmap=plt.cm.get_cmap('jet', len(np.unique(y))), s=10)
    plt.colorbar()
    plt.close(fig)
    # plt.show()
    fig.savefig(name)


def plot_confusion_matrix(predicted_labels, y_test, name, alphabet=None, xlabels=None):
    # draw confusion matrix
    # fig = plt.figure(figsize=(8, 6))
    plt.style.use('default')
    C = confusion_matrix(y_test, predicted_labels)
    conf_arr = C
    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                interpolation='nearest')
    width = len(conf_arr)
    height = len(conf_arr[0])

    for x in range(width):
        for y in range(height):
            ax.annotate(str(conf_arr[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')
    if xlabels != None:
        plt.xticks(range(width), xlabels[:width])
    if alphabet != None:
        plt.yticks(range(height), alphabet[:height])
    # plt.close(fig)
    plt.savefig(name)
    # plt.savefig('confusion_matrix.eps', format='eps')
    # plt.style.use('ggplot')


def KLdivergence(x, y):
    """Compute the Kullback-Leibler divergence between two multivariate samples.
    Parameters
    ----------
    x : 2D array (n,d)
      Samples from distribution P, which typically represents the true
      distribution.
    y : 2D array (m,d)
      Samples from distribution Q, which typically represents the approximate
      distribution.
    Returns
    -------
    out : float
      The estimated Kullback-Leibler divergence D(P||Q).
    References
    ----------
    Pérez-Cruz, F. Kullback-Leibler divergence estimation of
  continuous distributions IEEE International Symposium on Information
  Theory, 2008.
    """
    from scipy.spatial import cKDTree as KDTree

    # Check the dimensions are consistent
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    n, d = x.shape
    m, dy = y.shape

    assert(d == dy)

    # Build a KD tree representation of the samples and find the nearest neighbour
    # of each point in x.
    xtree = KDTree(x)
    ytree = KDTree(y)

    # Get the first two nearest neighbours for x, since the closest one is the
    # sample itself.
    r = xtree.query(x, k=2, eps=.01, p=2)[0][:, 1]
    s = ytree.query(x, k=1, eps=.01, p=2)[0]

    # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
    # on the first term of the right hand side.
    return -np.log(r/s).sum() * d / n + np.log(m / (n - 1.))


def plot_KLdivergence(recon_test, predicted_labels, num_digits, name):
    kl_matrix = np.zeros((num_digits, num_digits))
    for i in range(num_digits):
        class1 = recon_test[predicted_labels == i]
        for j in range(num_digits):
            class2 = recon_test[predicted_labels == j]
            kl_matrix[i, j] = KLdivergence(class1, class2)
    print(kl_matrix)
    plt.matshow(kl_matrix)
    plt.colorbar()
    tick_marks = np.arange(num_digits)
    plt.xticks(tick_marks, range(num_digits))
    plt.yticks(tick_marks, range(num_digits))
    plt.savefig(name)
    # plt.close()


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))

def myclassifiers(X_train, y_train, X_test, y_test, verbose=True):

    knn = KNeighborsClassifier(n_neighbors=1)
    neural_net = MLPClassifier(alpha=1, max_iter=1000)
    # gnb = GaussianNB()
    logistic = LogisticRegression(C=1e5)
    svc1 = svm.SVC(kernel="linear", C=0.025)
    svc2 = svm.SVC(kernel='rbf', gamma=.7, C=1.0)
    lda = LDA(solver='lsqr', shrinkage='auto')
    # qda = QDA()
    dTree = tree.DecisionTreeClassifier(max_depth=10)
    rForest = RandomForestClassifier(max_depth=10, n_estimators=20)
    adaBoost = AdaBoostClassifier()

    names = ["Nearest Neighbors(k=1)", "Neural Net", "Logistic", "Linear SVM", "RBF SVM", "LDA",
             "Decision Tree", "Random Forest", "AdaBoost"]
    classifiers = [knn, neural_net, logistic, svc1, svc2, lda, dTree, rForest, adaBoost]
    y_predict = []
    acc = []
#     print('Running',end="")
    for (i, clf) in enumerate(classifiers):
        if verbose:
            print(' %s... ' % names[i], end="")
        clf.fit(X_train, y_train)
        y_predict.append(clf.predict(X_test))
        classifier_score = clf.score(X_test, y_test)
        print(classifier_score)
        acc.append(classifier_score)
    return (acc, y_predict, names)


def myclassifiers2(X_train, y_train, X_test, y_test, verbose=True):

    # knn = KNeighborsClassifier(3)
    # gnb = GaussianNB()
    # neural_net = MLPClassifier(alpha=1, max_iter=1000)
    # svc1 = svm.SVC(kernel="linear", C=0.025)
    # svc2 = svm.SVC(gamma=2, C=1)
    # gpc = GaussianProcessClassifier(1.0 * RBF(1.0))
    # qda = QDA()
    # dTree = tree.DecisionTreeClassifier(max_depth=5)
    # rForest = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    # adaBoost = AdaBoostClassifier()

    names = ["Nearest Neighbors(k=3)", "Naive Bayes", "Neural Net", "Linear SVM","RBF SVM", "QDA",
             "Decision Tree", "Random Forest", "AdaBoost"]

    classifiers = [knn, gnb, neural_net, svc1, svc2, qda, dTree, rForest, adaBoost]
    y_predict = []
    acc = []
#     print('Running',end="")
    for (i, clf) in enumerate(classifiers):
        if verbose:
            print(' %s... ' % names[i], end="")
        clf.fit(X_train, y_train)
        y_predict.append(clf.predict(X_test))
        classifier_score = clf.score(X_test, y_test)
        print(classifier_score)
        acc.append(classifier_score)
    return (acc, y_predict, names)


# def find_majority(votes):
#     vote_count = Counter(votes)
#     top_two = vote_count.most_common(2)
#     if len(top_two)>1 and top_two[0][1] == top_two[1][1]:
#         # It is a tie
#         return -1
#     return top_two[0][0]

def find_majority(votes):
    vote_count = Counter(votes)
    top_two = vote_count.most_common(2)
    if len(top_two)>1 and top_two[0][1] == top_two[1][1]:
        # It is a tie
        return -1
    return top_two[0]


def find_meta(y_pred_test):
    new2 = np.vstack(y_pred_test).transpose()
    new3 = []
    for i in range(len(new2)):
        new3.append(find_majority(new2[i]))
    new4 = np.vstack(new3)
    return new4