### import collections
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import os, sys
from sklearn.datasets import make_classification, make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import pandas as pd
from collections import defaultdict
from sklearn import feature_extraction
from sklearn import preprocessing
from random import seed, shuffle
from onn import *
from cfsote import *
from load_law import *
from load_bank import *
from load_adult import *
from load_default import *
import argparse
# Initialize the argument parser
parser = argparse.ArgumentParser(description="pass the following arguments: dataset_name, number of clients, fairness notion,.")


# Add arguments
parser.add_argument("--fairness_notion", type=str, default='stp_score', 
                    choices=['eqop_score', 'stp_score'], 
                    help="Fairness notion to use. Options: 'eqop_score', 'stp_score'. Default is 'stp_score'.")
parser.add_argument("--num_clients", type=int, default=3, 
                    choices=[3, 5, 10, 15], 
                    help="Number of clients. Options: 3, 5, 10, 15. Default is 3.")
parser.add_argument("--dataset_name", type=str, default='bank', 
                    choices=['adult', 'default', 'bank', 'law'], 
                    help="Name of the dataset. Options: 'adult', 'default', 'bank', 'law'. Default is 'bank'.")
parser.add_argument("--distribution_type", type=str, default='random', 
                    choices=['random', 'attribute-based'], 
                    help="Data distribution type. Options: 'random', 'attribute-based'. Default is 'random'.")

# Parse the arguments
args = parser.parse_args()

# Store them in respective variables
num_clients = args.num_clients
fairness_notion = args.fairness_notion
dataset_name = args.dataset_name
distribution = args.distribution_type

if dataset_name == 'bank':
    if distribution == 'random':
        clients, client_index, client_window, client_window_label, client_eddm, length, p_Group, np_Group, sa_index = load_bank_random(num_clients)
    else:
        clients, client_index, client_window, client_window_label, client_eddm, length, p_Group, np_Group, sa_index = load_bank_attr()
elif dataset_name == 'default':
    if distribution == 'random':
        clients, client_index, client_window, client_window_label, client_eddm, length, p_Group, np_Group, sa_index = load_default_random(num_clients)
    else:
        clients, client_index, client_window, client_window_label, client_eddm, length, p_Group, np_Group, sa_index = load_default_attr()
elif dataset_name == 'law':
    if distribution == 'random':
        clients, client_index, client_window, client_window_label, client_eddm, length, p_Group, np_Group, sa_index = load_law_random(num_clients)
    else:
        clients, client_index, client_window, client_window_label, client_eddm, length, p_Group, np_Group, sa_index = load_law_attr()
elif dataset_name == 'adult':
    if distribution == 'random':
        clients, client_index, client_window, client_window_label, client_eddm, length, p_Group, np_Group, sa_index = load_adult_random(num_clients)
    else:
        clients, client_index, client_window, client_window_label, client_eddm, length, p_Group, np_Group, sa_index = load_adult_attr()
else:
    print("Dataset not supported, please add necessary piece of code for processing this data")

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from skmultiflow.drift_detection.eddm import EDDM
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight

num_clients = 3
#adwin = ADWIN(delta =1)
window=[]
window_label = []
window_warning = []
window_label_warning = []
pos_assigned=0
pos_samples = 0
neg_samples = 0
pos_syn_samples = 0
neg_syn_samples = 0
generated_samples_per_sample = 0
imbalance_ratio = 0 #of window
minority_label=1
majority_label = 0
lambda_initial=0.05  ###0.05 original
###
ocis = 0
classSize  = {}
class_weights_dict = {}
labels = []
###
j =0
change=0
warning=0

bal_acc_global = []  
disc_score_global = []  

eddm1 = EDDM()
eddm2 = EDDM()
eddm3 = EDDM()

#one global network
global_network = ONN(features_size=46, max_num_hidden_layers=5, qtd_neuron_per_hidden_layer=40, n_classes=2)

#3 networks for three clients
onn_network_1 = ONN(features_size=46, max_num_hidden_layers=5, qtd_neuron_per_hidden_layer=40, n_classes=2)
onn_network_2 = ONN(features_size=46, max_num_hidden_layers=5, qtd_neuron_per_hidden_layer=40, n_classes=2)
onn_network_3 = ONN(features_size=46, max_num_hidden_layers=5, qtd_neuron_per_hidden_layer=40, n_classes=2)

weight = 1
for _ in range(length):
    sum_w_output_layer, sum_b_output_layer, sum_w_hidden_layer, sum_b_hidden_layer = [],[],[],[]
    sum_alpha = []
    
    for (client_name, data) in clients.items():
        added_points = 0
        data, label = zip(*data)
        Y = np.asarray(label)
        X = np.asarray(data)
        
        if client_name=='client_1':
            eddm = eddm1
            onn_network = onn_network_1
        elif client_name=='client_2':
            eddm = eddm2
            onn_network = onn_network_2
        else:
            eddm = eddm3
            onn_network = onn_network_3
            
        i = client_index[client_name]
        if i ==0:
            print(client_index)
        else:
            if i%200==0:
                galpha, gw_output_layer, gb_output_layer, gw_hidden_layer, gb_hidden_layer = global_network.get_weights('global')
                onn_network.set_weights(galpha, gw_output_layer, gb_output_layer, gw_hidden_layer, gb_hidden_layer)
        
        if np.size(client_window[client_name])!=0:
            if client_window[client_name].ndim==2 and len(client_window[client_name])>30:
                majority_count = list(client_window_label[client_name]).count(0)
                minority_count = list(client_window_label[client_name]).count(1)
                if majority_count >  minority_count and minority_count!=0:
                    weight = int(majority_count/minority_count)
        if Y[i]==minority_label:
            onn_network.partial_fit(np.asarray([X[i, :]]), np.asarray([Y[i]]),weight)
            
        else:
            onn_network.partial_fit(np.asarray([X[i, :]]), np.asarray([Y[i]]),1)
        
        
        
        #onn_network.partial_fit(np.asarray([X[i, :]]), np.asarray([Y[i]]))
    
        if np.size(client_window[client_name])==0:
            client_window[client_name] = np.array(X[i])
            client_window_label[client_name] = np.array(Y[i])
        else:
            client_window[client_name]=np.vstack((client_window[client_name],np.array(X[i])))
            client_window_label[client_name]= np.vstack((client_window_label[client_name],np.array(Y[i])))
        eddm.add_element(Y[i])
    

        if eddm.detected_change():
            print('Change has been detected in data: ' + str(Y[i]) + ' - of index: ' + str(i))
            change+=1
            client_window[client_name] = []
            client_window_label[client_name] = []
            onn_network.reset_eval_metrics()
            
        pos_assigned = onn_network.tp+onn_network.fp-0.2
        pos_samples = onn_network.tp+onn_network.fn-0.2
        if fairness_notion == 'eqop':
            disc_notion = onn_network.eqop_score
        else:
            disc_notion = onn_network.stp_score
        if np.size(client_window[client_name])!=0:
            if client_window[client_name].ndim==2 and len(client_window[client_name])>30:
                if disc_notion < 0:
                    pp_Group = np_Group
                    npp_Group = p_Group
                else:
                    pp_Group = p_Group
                    npp_Group = np_Group
                disc_score = abs(disc_notion)    
                if disc_score > 0.01:
                    #print(onn_network.stp_score)
                    lambda_score = lambda_initial*(1+(disc_score/0.1))
                    if pos_assigned <= pos_samples:
                        X_syn,Y_syn = create_synth_data(client_window[client_name], client_window_label[client_name], minority_label,majority_label,4,lambda_score, 'min_p', pp_Group,npp_Group)
                    else:
                        X_syn,Y_syn = create_synth_data(client_window[client_name], client_window_label[client_name], minority_label,majority_label,4,lambda_score, 'maj_np', pp_Group,npp_Group)
                    if X_syn!=-1:
                        Y_syn = np.array(Y_syn)
                        X_syn = np.array(X_syn)
                        X_syn, Y_syn = shuffle(X_syn, Y_syn, random_state=0)
                        added_points = len(Y_syn)
                        for k in range(len(X_syn)):
                            onn_network.partial_fit(np.asarray([X_syn[k, :]]), np.asarray([Y_syn[k]]), 1, test='no')
        client_alpha, client_w_output_layer, client_b_output_layer, client_w_hidden_layer, client_b_hidden_layer = onn_network.get_weights('client')
        i=i+1
        client_index.update({client_name:i})
        #scaling_factor = (client_index[client_name]+added_points)/(sum(client_index.values())+added_points)
        scaling_factor = 1/5
        print(client_index)
        p = 0
        if p==0:
            if sum_alpha==[]:
                sum_alpha = torch.mul(client_alpha, scaling_factor)
                sum_w_output_layer = client_w_output_layer
                sum_b_output_layer = client_b_output_layer
                sum_w_hidden_layer = client_w_hidden_layer
                sum_b_hidden_layer = client_b_hidden_layer
            
                for j in range(onn_network.max_num_hidden_layers):
                    sum_w_output_layer[j] = torch.mul(client_w_output_layer[j], scaling_factor)
                    sum_b_output_layer[j] = torch.mul(client_b_output_layer[j], scaling_factor)
                    sum_w_hidden_layer[j] = torch.mul(client_w_hidden_layer[j], scaling_factor)
                    sum_b_hidden_layer[j] = torch.mul(client_b_hidden_layer[j], scaling_factor)

            else:
                sum_alpha = torch.add(sum_alpha, torch.mul(client_alpha, scaling_factor))
                for j in range(onn_network.max_num_hidden_layers):
                    sum_w_output_layer[j] = torch.add(sum_w_output_layer[j],torch.mul(client_w_output_layer[j], scaling_factor)) 
                    sum_b_output_layer[j] = torch.add(sum_b_output_layer[j],torch.mul(client_b_output_layer[j], scaling_factor))
                    sum_w_hidden_layer[j] = torch.add(sum_w_hidden_layer[j],torch.mul(client_w_hidden_layer[j], scaling_factor)) 
                    sum_b_hidden_layer[j] = torch.add(sum_b_hidden_layer[j],torch.mul(client_b_hidden_layer[j], scaling_factor))
        
    if i%200==0:    
        global_network.set_weights(sum_alpha, sum_w_output_layer, sum_b_output_layer, sum_w_hidden_layer, sum_b_hidden_layer)

        x_test, y_test = shuffle(x_test, y_test, random_state=0)
        
        for m in range(len(x_test)-1):
            prediction_1 = global_network.predict_1(np.asarray([x_test[m, :]]))
            global_network.update_eval_metrics(prediction_1,np.asarray([y_test[m]]))
            global_network.update_stp_score(prediction_1,np.asarray([x_test[m, :]]))
            global_network.update_eqop_score(prediction_1,np.asarray([x_test[m, :]]),np.asarray([y_test[m]]))
        
        bal_acc_global.append(global_network.bal_acc)
        disc_score_global.append(global_network.eqop_score)
        global_network.reset_eval_metrics()
global_network.set_weights(sum_alpha, sum_w_output_layer, sum_b_output_layer, sum_w_hidden_layer, sum_b_hidden_layer) 
for n in range(len(x_test)-1):
    prediction_1 = global_network.predict_1(np.asarray([x_test[n, :]]))
    global_network.update_eval_metrics(prediction_1,np.asarray([y_test[n]]))
    global_network.update_stp_score(prediction_1,np.asarray([x_test[n, :]]))
    global_network.update_eqop_score(prediction_1,np.asarray([x_test[n, :]]),np.asarray([y_test[n]]))

bal_acc_global.append(global_network.bal_acc)  
disc_score_global.append(global_network.eqop_score)  
        
        #print("change" + str(change))
#print("warning" + str(warning))
    
    
    
    
    
print("Balanced accuracy: " + str(onn_network.bal_acc))
print("Sensitivity: " + str(onn_network.sen))
print("Specificity: " + str(onn_network.spec))
print("Stp score: " + str(onn_network.stp_score))
print("Eqop score: " + str(onn_network.eqop_score))
