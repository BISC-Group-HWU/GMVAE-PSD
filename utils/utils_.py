"""
FINAL TESTS PSD
"""

"""
You can augment the data with translations to be translation-invariant

"""

# import tkinter
# import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
from sklearn.metrics import confusion_matrix
import numpy as np
from numba import jit

# def Get_PSD2_SIMULATED_UNNORMALISED(data,labels):

# #### PSD settings #####################################################################################################
#     tail_end=183
#     tail_start=11
#     total_start = 2

# #### PSD Calculation and save the neutron/gamma label #################################################################
#     for i in range(0, len(data)):
#         if labels[i] == 0:
#             max_ch = np.argmax(data[i])
#             total=sum(data[i,max_ch-total_start:max_ch-total_start+185])
#             tail=sum(data[i,max_ch+tail_start:max_ch+tail_end])
#             ttl=tail/total
#             discrim_number=tail-(-0.032605*total*total+0.32424*total+0.0024835)
#             if (discrim_number>0): #Neutrons
#                 labels[i] = 1
#     return labels

def Get_PSD_MEASURED(data,labels):

#### PSD settings #####################################################################################################
    tail_end=183  # the tail integration ends at 150 sample
    tail_start=9   # number of samples after the peak to start the tail integration
    total_start = 2

    for i in range(0, len(data)):
        if labels[i] == 0:
            max_ch = np.argmax(data[i])
            total=sum(data[i,max_ch-total_start:max_ch-total_start+185])
            tail=sum(data[i,max_ch+tail_start:max_ch+tail_end])
            if (total<1.9):
                Stilbene_discrim_number=tail-(-0.032605*total*total+0.32424*total+0.0024835)
            else:
                Stilbene_discrim_number=tail-(0.22898*total+0.065817)
            if (Stilbene_discrim_number>0): #Neutrons
                    labels[i] = 1

    print(max_ch,total,tail)
    return labels

@jit
def peak_finding_strategy_real_data(p):
    """Based on Ming's piled-up rejection code.
    
    Args:
        p (np.ndarray): Voltage pulse

    Returns:
        list[int]: The indices of peaks found. Returns immediately when it detects the second peak.
    """
    #### Ming's piled-up rejection code.
    #### 2633 piled-up pulses detected
    risingEdgeWidth = 6 # samples, width of pulse rising edge
    peakRatioThreshold = 0.1 #0.03 #0.1 second peak height / major peak height > 0.1. Decrease this will results in more pulses being classified as piled-up.
    peakHeightThreshold = 0.005 #0.008 #0.005 second peak height > 5 mV.  Decrease this will results in more pulses being classified as piled-up.

    # peakRatioThreshold = 0.03 #0.1 # second peak height / major peak height > 0.1. Decrease this will results in more pulses being classified as piled-up.
    # peakHeightThreshold = 0.008 #0.005 # second peak height > 5 mV.  Decrease this will results in more pulses being classified as piled-up.

    peakIndex = np.argmax(p)
    peakHeight = p[peakIndex]
    for i in range(peakIndex-2*risingEdgeWidth):
        # detect the rising edge of the secondary pulse before the major peak
        deltaV = p[i+risingEdgeWidth] - p[i]
        if deltaV > peakRatioThreshold * peakHeight and deltaV > peakHeightThreshold:
            return [i+risingEdgeWidth, peakIndex]
    for i in range(peakIndex, len(p) - risingEdgeWidth):
        # detect the rising edge of the secondary pulse after the major peak
        deltaV = p[i+risingEdgeWidth] - p[i]
        if deltaV > peakRatioThreshold * peakHeight and deltaV > peakHeightThreshold:
            return [peakIndex, i+risingEdgeWidth]
    return [peakIndex]

def reject_pile_ups_labels_Ming(pulses):
    labels_new = np.zeros((len(pulses)))
    for i in range(len(pulses)):
        ind = peak_finding_strategy_real_data(pulses[i])
        if len(ind) > 1:
            labels_new[i] = 2
    return labels_new

def split_data(x_gammas,num_data,num_train):
     np.random.seed(0)
     indices = np.random.permutation(len(x_gammas))
     x_gammas = x_gammas[indices[:num_data]]
     x_gammas_train = x_gammas[:num_train]
     x_gammas_test = x_gammas[num_train:]
     return x_gammas_train, x_gammas_test


def split_train_test(gamma_pulses_clean,neutron_pulses_clean,train_size = 5000):
    np.random.seed(0)
    indices_gammas = np.random.permutation(len(gamma_pulses_clean))
    np.random.seed(0)
    indices_neutrons = np.random.permutation(len(neutron_pulses_clean))
    
    train_gammas, train_neutrons = indices_gammas[:train_size], indices_neutrons[:train_size]
    test_gammas, test_neutrons = indices_gammas[train_size:], indices_neutrons[train_size:]

    x_train = np.append(gamma_pulses_clean[train_gammas],neutron_pulses_clean[train_neutrons],axis=0)
    y_train = np.append([0]*train_size,[1]*train_size,axis=0) 

    x_test = np.append(gamma_pulses_clean[test_gammas],neutron_pulses_clean[test_neutrons],axis=0)
    y_test = np.append([0]*len(gamma_pulses_clean[test_gammas]),[1]*len(neutron_pulses_clean[test_neutrons]),axis=0)

    return x_train,y_train,x_test,y_test 

def normalise(x_train, norm_type=None):
    x_train_copy = np.copy(x_train)  # Create a copy of the input array
    for i in range(len(x_train_copy)):
        if norm_type == 'fft':
            x_train_copy[i] = np.abs(np.fft.fftshift(np.fft.fft(x_train_copy[i])))
            x_train_copy[i] = x_train_copy[i] / max(x_train_copy[i])
        elif norm_type == 'max':
            x_train_copy[i] = x_train_copy[i] / max(x_train_copy[i])
    return x_train_copy

def add_noise(xt0,sig):
    np.random.seed(0)
    return  xt0 + sig*np.random.randn(xt0.shape[0],xt0.shape[1])

def addnoise_multi(xt0,y,sig,seed_i):
    num_classes = len(np.unique(y))
    num_sigmas = len(sig)
    sample_width = xt0.shape[1]
    
    # get number of labeled data per class (balanced partition)
    for i in range(num_classes):
        indices = np.where(y == i)[0]
        labeled_per_class = len(indices)//num_sigmas
        for j in range(num_sigmas):
            np.random.seed(seed_i)
            xt0[indices[j*labeled_per_class:j*labeled_per_class+labeled_per_class]] += sig[j]*np.random.randn(labeled_per_class,sample_width)
    return  xt0

    