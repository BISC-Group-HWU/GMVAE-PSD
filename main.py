"""
This is the main code to generate the results for 2classes dataset from Angela
GMVAE

"""
import os 
learn_type = 'semisupervised'
hidden1 = 148
hidden2 = 74
# feature_dim = 50   ## check where this apply 
latent_dim = 37
num_epoch = 100
batch_size = 200

test_trusted1 = 1
test_trusted2 = 1

num_labeled = 1

sig = 5e-2
num_classes = 3
xlabels = ['0', '1', '2']
class_type = ['Gamma', 'Neutron', 'Pile-up']

num_data = 50000 
num_train = 40000

num_train1 = 10000
num_test1 = 2500

load_model = 1

import tkinter
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from utils.partition import *
from utils.utils_ import *
from model.SSVAE import *
from scipy.io import loadmat, savemat
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# from utils.plots import *
import sklearn
import scipy
import hickle as hkl
from scipy.optimize import curve_fit
import time

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)
del_all_flags(tf.flags.FLAGS)

flags = tf.flags
PARAMETERS = flags.FLAGS

#########################################################
## Input Parameters
#########################################################

## Dataset
flags.DEFINE_string('dataset', 'PLAID', 'Specify the desired dataset (mnist, usps, reuters10k)')
flags.DEFINE_integer('seed', -1, 'Random Seed')

## GPU
flags.DEFINE_integer('gpu', 1, 'Using Cuda, 1 to enable')
flags.DEFINE_integer('gpuID', 2, 'Set GPU Id to use')

## Training
flags.DEFINE_integer('batch_size', batch_size, 'Batch size of training data')
flags.DEFINE_integer('num_epochs', num_epoch,
                     'Number of epochs in training phase')
flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate for training')
flags.DEFINE_integer('pretrain', 0, 'Number of epochs to train the model without the metric loss')
flags.DEFINE_float('decay_epoch', -1, 'Reduces the learning rate every decay_epoch')
flags.DEFINE_float('lr_decay', 0.5, 'Learning rate decay for training')

## Architecture
flags.DEFINE_integer('num_neurons_big', hidden1, 'Number of neurons big')
flags.DEFINE_integer('num_neurons_small', hidden2, 'Number of neurons small')
flags.DEFINE_integer('num_classes', num_classes, 'Number of clusters')
# flags.DEFINE_integer('feature_size', feature_dim, 'Size of the hidden layer that splits gaussian and categories')
flags.DEFINE_integer('gaussian_size', latent_dim,
                     'Size of the gaussian learnt by the network')

## Partition parameters
flags.DEFINE_float('train_proportion', 0.9, '(0.9144) Proportion of examples to consider for training only  (0.0-1.0)')
flags.DEFINE_integer('batch_size_val', 200, 'Batch size of validation data')
flags.DEFINE_integer('batch_size_test', 200, 'Batch size of test data')

## Gumbel parameters
flags.DEFINE_float('temperature', 1.0, 'Initial temperature used in gumbel-softmax (recommended 0.5-1.0)')
flags.DEFINE_integer('decay_temperature', 1, 'Set 1 to decay gumbel temperature at every epoch')
flags.DEFINE_integer('hard_gumbel', 0, 'Hard version of gumbel-softmax')
flags.DEFINE_float('min_temperature', 0.5, 'Minimum temperature of gumbel-softmax after annealing' )
flags.DEFINE_float('decay_temp_rate', 0.013862944, 'Temperature decay rate at every epoch')

## Loss function parameters
flags.DEFINE_string('loss_type', 'bce', 'Desired loss function to train (mse, bce)')
flags.DEFINE_float('w_gaussian', 1.0, 'Weight of Gaussian regularization')
flags.DEFINE_float('w_categorical', 1.0, 'Weight of Categorical regularization')
flags.DEFINE_float('w_reconstruction', 1.0, 'Weight of Reconstruction loss')
flags.DEFINE_float('w_metric', 1.0, 'Weight of metric distance loss')
flags.DEFINE_float('w_assign', 1.0, 'Weight of assignment loss')
flags.DEFINE_float('metric_margin', 0.5, 'Margin of metric loss')
flags.DEFINE_string('metric_loss', 'triplet', 'Desired metric loss function to train (triplet, lifted)')
flags.DEFINE_float('anneal_w_metric', 0, 'Set 1 to anneal metric loss weight every epoch')

## Semisupervised
flags.DEFINE_integer('num_labeled', num_labeled, 'Number of labeled data to consider in training')
# flags.DEFINE_integer('num_labeled_batch', batch_size_labeled, 'Number of labeled data to consider in training')

## Others
flags.DEFINE_integer('verbose', 1, "Print extra information at every epoch.")
flags.DEFINE_integer('random_search_it', 20, 'Iterations of random search')

if PARAMETERS.gpu == 1:
   os.environ["CUDA_VISIBLE_DEVICES"] = str(PARAMETERS.gpuID)

if PARAMETERS.seed < 0:
   np.random.seed(None)
else:
   np.random.seed(PARAMETERS.seed)


#########################################################
## Read Data
#########################################################
# mat = loadmat("/home/aa587/codes/VAE/Python/PSD-Angela/Data/Simulated_pulse_from_Matlab/pile-up pulses 2/simulated_pulses.mat")
mat = loadmat("./Simulated_pulse_from_Matlab/pile-up pulses 2/simulated_pulses.mat")
gamma_pulses = mat['gamma_pulses']
neutron_pulses = mat['neutron_pulses']
pp_pulses = mat['pp_pulses']
pn_pulses = mat['pn_pulses']
np_pulses = mat['np_pulses']
nn_pulses = mat['nn_pulses']

# CHECK IF WE NEED NORMALISATION
gamma_pulses_normalised = normalise(gamma_pulses,'max')
neutron_pulses_normalised = normalise(neutron_pulses,'max')
pp_pulses_normalised = normalise(pp_pulses,'max')
pn_pulses_normalised = normalise(pn_pulses,'max')
np_pulses_normalised = normalise(np_pulses,'max')
nn_pulses_normalised = normalise(nn_pulses,'max')

x_gammas = add_noise(gamma_pulses_normalised,sig)
x_neutrons = add_noise(neutron_pulses_normalised,sig)
x_pp = add_noise(pp_pulses_normalised,sig)
x_pn = add_noise(pn_pulses_normalised,sig)
x_np = add_noise(np_pulses_normalised,sig)
x_nn = add_noise(nn_pulses_normalised,sig)

x_gammas_train, x_gammas_test = split_data(x_gammas,num_data,num_train)
x_neutrons_train, x_neutrons_test = split_data(x_neutrons,num_data,num_train)
x_pp_train, x_pp_test = split_data(x_pp,num_data,num_train1)
x_pn_train, x_pn_test = split_data(x_pn,num_data,num_train1)
x_np_train, x_np_test = split_data(x_np,num_data,num_train1)
x_nn_train, x_nn_test = split_data(x_nn,num_data,num_train1)

x_pileup_train = np.vstack([x_pp_train,x_pn_train,x_np_train,x_nn_train])
x_pileup_test = np.vstack([x_pp_test,x_pn_test,x_np_test,x_nn_test])

x_train = np.vstack([x_gammas_train,x_neutrons_train,x_pileup_train])
y_train = [0] * len(x_gammas_train) + [1] * len(x_neutrons_train) + [2] * len(x_pileup_train)
y_train = np.hstack(y_train)

# LOAD NEW GAMMAS AND NEUTRONS DATA
gamma_pulses2 = np.loadtxt("./Simulated_pulse_from_Matlab/Gamma_pulse_2.txt", dtype='f', delimiter=' ')
gamma_pulses2 = gamma_pulses2.reshape((-1,296))
neutron_pulses2 = np.loadtxt("./Simulated_pulse_from_Matlab/Neutron_pulse_2.txt", dtype='f', delimiter=' ')
neutron_pulses2 = neutron_pulses2.reshape((-1,296))
gamma_pulses2 = normalise(gamma_pulses2,'max')
neutron_pulses2 = normalise(neutron_pulses2,'max')
gamma_pulses2 = add_noise(gamma_pulses2,sig)
neutron_pulses2 = add_noise(neutron_pulses2,sig)

x_gammas_test = np.append(x_gammas_test,gamma_pulses2,axis=0)
x_neutrons_test = np.append(x_neutrons_test,neutron_pulses2,axis=0)

x_test1 = np.vstack([x_gammas_test,x_neutrons_test,x_pileup_test])
y_test1 = [0] * len(x_gammas_test) + [1] * len(x_neutrons_test) + [2] * len(x_pileup_test)
y_test1 = np.hstack(y_test1)

## Set datatypes
x_train = x_train.astype(np.float32)
x_train = flatten_array(x_train)
y_train = y_train.astype(np.int64)

x_test1 = x_test1.astype(np.float32)
x_test1 = flatten_array(x_test1)
y_test1 = y_test1.astype(np.int64)

# print dataset shape
print('Train size: ', x_train.shape, ' Test size: ', x_test1.shape)
print('Train size: ', y_train.shape, ' Test size: ', y_test1.shape)

# --------------------------------------------------------------
# --------------------------------------------------------------
# LOAD TEST DATA - TRUSTED 1 (SIMULATED DATA) - PSD 
# --------------------------------------------------------------
# --------------------------------------------------------------
mat = loadmat("./Simulated_pulse_from_Matlab/pile-up pulses 2/simulated_pulses.mat")
gamma_pulses = mat['gamma_pulses']
neutron_pulses = mat['neutron_pulses']
pp_pulses = mat['pp_pulses']
pn_pulses = mat['pn_pulses']
np_pulses = mat['np_pulses']
nn_pulses = mat['nn_pulses']

_, x_gammas_test = split_data(gamma_pulses,num_data,num_train)
_, x_neutrons_test = split_data(neutron_pulses,num_data,num_train)

# LOAD NEW GAMMAS AND NEUTRONS DATA
gamma_pulses2 = np.loadtxt("./Simulated_pulse_from_Matlab/Gamma_pulse_2.txt", dtype='f', delimiter=' ')
gamma_pulses2 = gamma_pulses2.reshape((-1,296))
neutron_pulses2 = np.loadtxt("./Simulated_pulse_from_Matlab/Neutron_pulse_2.txt", dtype='f', delimiter=' ')
neutron_pulses2 = neutron_pulses2.reshape((-1,296))

x_gammas_test = np.append(x_gammas_test,gamma_pulses2,axis=0)
x_neutrons_test = np.append(x_neutrons_test,neutron_pulses2,axis=0)

_, x_pp_test = split_data(pp_pulses,num_data,num_train1)
_, x_pn_test = split_data(pn_pulses,num_data,num_train1)
_, x_np_test = split_data(np_pulses,num_data,num_train1)
_, x_nn_test = split_data(nn_pulses,num_data,num_train1)
x_pileup_test = np.vstack([x_pp_test,x_pn_test,x_np_test,x_nn_test])

x_test1_clean = np.vstack([x_gammas_test,x_neutrons_test,x_pileup_test])
y_test1_clean = [0] * len(x_gammas_test) + [1] * len(x_neutrons_test) + [2] * len(x_pileup_test)
y_test1_clean = np.hstack(y_test1_clean)

x_test1_clean = x_test1_clean.astype(np.float32)
x_test1_clean = flatten_array(x_test1_clean)
y_test1_clean = y_test1_clean.astype(np.int64)

#########################################################
## Data Partition
#########################################################
train_data, train_labels, val_data, val_labels = x_train, y_train, x_test1[:1000], y_test1[:1000]

batch_size_labeled = 99
# batch_size_labeled = 30
flags.DEFINE_integer('num_labeled_batch', batch_size_labeled, 'Number of labeled data to consider in training')

flags.DEFINE_integer('knn', 1, 'Number of k-nearest-neighbors to consider in assignment')


if PARAMETERS.verbose == 1:
   print("Train size: %dx%d" % (train_data.shape[0], train_data.shape[1]))
   if PARAMETERS.train_proportion < 1.0:
      print("Validation size: %dx%d" % (val_data.shape[0], val_data.shape[1]))
   print("Test size: %dx%d" % (x_test1.shape[0], x_test1.shape[1]))
   

#########################################################
## Train and Test Model
#########################################################
# works perfectly
metric_loss = 'triplet'
PARAMETERS.w_assign = 50
PARAMETERS.w_reconstruction = 50
PARAMETERS.w_metric = 50
PARAMETERS.w_categorical = 200
PARAMETERS.w_gaussian = 1

# Obtain labeled data
# num_labeled = [99, 300, 600, 900, 3000, 6000, 9000, 120000]
num_labeled = [99]

# save_path1 = './results_addnoise_after_normalise_PAPER/sig='+str(sig)+'/'
save_path1 = '.'
if not os.path.exists(save_path1):
     os.mkdir(save_path1)

seed_i = 16
for  seed_i in range(16,17):
	for num_labeled_i in num_labeled: 
                        #  print('seed='+str(seed_i))
                         PARAMETERS.num_labeled = num_labeled_i
                         _, _, labeled_data, labeled_labels, _, _ = create_semisupervised_dataset(train_data, train_labels, PARAMETERS.num_classes, num_labeled_i, seed_i)


                         save_path = save_path1
                         if not os.path.exists(save_path):
                              os.mkdir(save_path)

                         if load_model:
                              save_dir = save_path+'/checkpoints/'
                         else:
                              save_dir = save_path+'/checkpoints/'
                              if not os.path.exists(save_dir):
                                   os.mkdir(save_dir)
                      
                         name_save = save_path+'/PSD'+'_w_assign='+str(PARAMETERS.w_assign)+'_w_recon='+str(PARAMETERS.w_reconstruction)+'_w_metric='+str(PARAMETERS.w_metric)+'_w_cat='+str(PARAMETERS.w_categorical)+'_w_kl='+str(PARAMETERS.w_gaussian)+'_data='+str(num_train)+'_classes='+str(num_classes)+'_epoch='+str(num_epoch)+'_latent='+str(latent_dim)+'_sig='+str(sig)+'_metric='+metric_loss #+'_FINAL'

                         if load_model == 0:
                              tf.reset_default_graph()
                              if PARAMETERS.seed > -1:
                                   tf.set_random_seed(PARAMETERS.seed)
                              ## Model Initialization
                              vae = SSVAE(PARAMETERS)
                              ## Training Phase
                              history_loss = vae.train(train_data, train_labels, val_data, val_labels, labeled_data, labeled_labels,load_model,save_dir)
                              logits, predicted_labels, accuracy, nmi, avg_accuracy = vae.test(train_data, train_labels, PARAMETERS.batch_size_test)

                         if load_model:
                              tf.reset_default_graph()
                              if PARAMETERS.seed > -1:
                                   tf.set_random_seed(PARAMETERS.seed)
                              ## Model Initialization
                              vae = SSVAE(PARAMETERS)
                              ## Loading Phase
                              history_loss = vae.train(train_data, train_labels, val_data, val_labels, labeled_data, labeled_labels,load_model,save_dir)
                              logits, predicted_labels, accuracy, nmi, avg_accuracy = vae.test(train_data, train_labels, PARAMETERS.batch_size_test)

                         if test_trusted1:
                              ## Testing Phase
                              xdata, ylabels = x_test1, y_test1
                              logits_test, predicted_labels_test, accuracy_test, nmi, avg_accuracy_test = vae.test(xdata, ylabels, PARAMETERS.batch_size_test)
                              ind_test, counts_test = np.unique(predicted_labels_test, return_counts=True)
                              acc = sklearn.metrics.accuracy_score(ylabels, predicted_labels_test)

print("Done!")

