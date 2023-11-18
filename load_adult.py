import torch
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from psmpy import PsmPy
from psmpy.functions import cohenD
from psmpy.plotting import *
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import preprocessing
from random import seed, shuffle
import random
def create_clients(instances, labels, num_clients, initial='clients'):
    ''' return: a dictionary with keys clients' names and value as 
                data shards - tuple of images and label lists.
        args: 
            image_list: a list of numpy arrays of training images
            label_list:a list of binarized labels for each image
            num_client: number of fedrated members (clients)
            initials: the clients'name prefix, e.g, clients_1 
            
    '''

    #create a list of client names
    client_names = ['{}_{}'.format(initial, i+1) for i in range(num_clients)]

    #randomize the data
    data = list(zip(instances, labels))
    random.shuffle(data)

    #shard data and place at each client
    size = len(data)//num_clients
    shards = [data[i:i + size] for i in range(0, size*num_clients, size)]

    #number of clients must equal number of shards
    assert(len(shards) == len(client_names))

    return {client_names[i] : shards[i] for i in range(len(client_names))} 

def load_adult_random(num_clients):
    FEATURES_CLASSIFICATION = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation",
                               "relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country"] #features to be used for classification
    CONT_VARIABLES = ["age","fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"] # continuous features, will need to be handled separately from categorical features, categorical features will be encoded using one-hot
    CLASS_FEATURE = "y" # the decision variable
    SENSITIVE_ATTRS = ["sex"]
    
    COMPAS_INPUT_FILE = "./datasets/adult2.csv"

    # load the data and get some stats
    df = pd.read_csv(COMPAS_INPUT_FILE)
    df.dropna()
    # convert to np arra
    data = df.to_dict('list')

    for k in data.keys():
        data[k] = np.array(data[k])

    """ Feature normalization and one hot encoding """
   
    y = data[CLASS_FEATURE]
    #y[y==0] = -1
    X = np.array([]).reshape(len(y), 0) # empty array with num rows same as num examples, will hstack the features to it
    x_control = defaultdict(list)
    
    feature_names = []
    for attr in FEATURES_CLASSIFICATION:
        vals = data[attr]
        if attr in CONT_VARIABLES:
            vals = [float(v) for v in vals]
            vals = preprocessing.scale(vals) # 0 mean and 1 variance
            vals = np.reshape(vals, (len(y), -1)) # convert from 1-d arr to a 2-d arr with one col
        
        else: # for binary categorical variables, the label binarizer uses just one var instead of two
            lb = preprocessing.LabelBinarizer()   #LabelEncoder() # Label Encoder
            lb.fit(vals)
            vals = lb.transform(vals)
            #vals = np.reshape(vals, (len(y), -1))
            #if attr =="sex": 
            #    print(lb.classes_)
            #    print(lb.transform(lb.classes_))
        
        # add to sensitive features dict
        if attr in SENSITIVE_ATTRS:
            x_control[attr] = vals

        # add to learnable features
        X = np.hstack((X, vals))

        if attr in CONT_VARIABLES: # continuous feature, just append the name
            feature_names.append(attr)
        else: # categorical features
            if vals.shape[1] == 1: # binary features that passed through lib binarizer
                feature_names.append(attr)
            else:
                for k in lb.classes_: # non-binary categorical features, need to add the names for each cat
                    feature_names.append(attr + "_" + str(k))


    # convert the sensitive feature to 1-d array
    x_control = dict(x_control)
    for k in x_control.keys():
        assert(x_control[k].shape[1] == 1) # make sure that the sensitive feature is binary after one hot encoding
        x_control[k] = np.array(x_control[k]).flatten()
    p_Group = 0
    np_Group = 1
    sa_index = feature_names.index(SENSITIVE_ATTRS[0])
    clients = create_clients(X, y, num_clients, initial='client')
    client_index = {}
    client_window = {}
    client_window_label = {}
    client_eddm = {}

    for (client_name, data) in clients.items():
        data, label = zip(*data)
        Y = np.asarray(label)
        X = np.asarray(data)
        client_index.update({client_name:0})
        client_window.update({client_name:[]})
        client_window_label.update({client_name:[]})
        length = len(data)
    return clients, client_index, client_window, client_window_label, client_eddm, length, p_Group, np_Group, sa_index

def create_clients_attr(Xtr1, Ytr1,Xtr2,Ytr2,Xtr3,Ytr3,num_clients,initial='clients'):
    ''' return: a dictionary with keys clients' names and value as 
                data shards - tuple of images and label lists.
        args: 
            image_list: a list of numpy arrays of training images
            label_list:a list of binarized labels for each image
            num_client: number of fedrated members (clients)
            initials: the clients'name prefix, e.g, clients_1 
            
    '''
    clients = {}
    #create a list of client names
    client_names = ['{}_{}'.format(initial, i+1) for i in range(num_clients)]
    
    data = list(zip(Xtr1, Ytr1))
    random.shuffle(data)
    
    clients.update({client_names[0] :data})
    data = list(zip(Xtr2, Ytr2))
    clients.update({client_names[1] :data})
    data = list(zip(Xtr3, Ytr3))
    clients.update({client_names[2] :data})
    #data = list(zip(Xtr4, Ytr4))
    #clients.update({client_names[3] :data})

    return clients
def load_adult_attr():
    FEATURES_CLASSIFICATION = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation",
                               "relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country"] #features to be used for classification
    CONT_VARIABLES = ["age","fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"] # continuous features, will need to be handled separately from categorical features, categorical features will be encoded using one-hot
    CLASS_FEATURE = "y" # the decision variable
    SENSITIVE_ATTRS = ["sex"]
    
    COMPAS_INPUT_FILE = "./datasets/adult2.csv"
    CAT_VARIABLES_INDICES = [1,2,3,4,6,7,8,10,14,15]
    

    df = pd.read_csv(COMPAS_INPUT_FILE)
    
    # convert to np array
    
    data = df.to_dict('list')
    
    for k in data.keys():
        data[k] = np.array(data[k])

    """ Feature normalization and one hot encoding """

    # convert class label 0 to -1
    y = data[CLASS_FEATURE]
    #y[y == "yes"] = 1
    #y[y == 'no'] = 0
    y = np.array([int(k) for k in y])

    X = np.array([]).reshape(len(y), 0)  # empty array with num rows same as num examples, will hstack the features to it
    
    x_control = defaultdict(list)
    i=0
    feature_names = []
    
    for attr in FEATURES_CLASSIFICATION:
        vals = data[attr]
        
        if attr in CONT_VARIABLES:
            vals = [float(v) for v in vals]
            vals = preprocessing.scale(vals)  # 0 mean and 1 variance
            vals = np.reshape(vals, (len(y), -1))  # convert from 1-d arr to a 2-d arr with one col
            

        else:  # for binary categorical variables, the label binarizer uses just one var instead of two
            lb = preprocessing.LabelBinarizer()
            lb.fit(vals)
            vals = lb.transform(vals)
            #vals = np.reshape(vals, (len(y), -1))
           
            
        # add to sensitive features dict
        if attr in SENSITIVE_ATTRS:
            x_control[attr] = vals
            

        # add to learnable features
        X = np.hstack((X, vals))
        
        if attr in CONT_VARIABLES:  # continuous feature, just append the name
            feature_names.append(attr)
        else:  # categorical features
            if vals.shape[1] == 1:  # binary features that passed through lib binarizer
                feature_names.append(attr)
            else:
                for k in lb.classes_:  # non-binary categorical features, need to add the names for each cat
                    feature_names.append(attr + "_" + str(k))

    # convert the sensitive feature to 1-d array
    
    x_control = dict(x_control)
    
    for k in x_control.keys():
        assert (x_control[k].shape[1] == 1)  # make sure that the sensitive feature is binary after one hot encoding
        x_control[k] = np.array(x_control[k]).flatten()
    
    feature_names.append('target')
    p_Group = 0
    np_Group = 1
    sa_index = feature_names.index(SENSITIVE_ATTRS[0])

    age_group_1 = []
    age_group_2 = []
    age_group_3 = []
    for i in range(len(df)):
        if df['age'].iloc[i]>=0 and df['age'].iloc[i]<30:
            age_group_1.append(i)
        elif df['age'].iloc[i]>29 and df['age'].iloc[i]<40:
            age_group_2.append(i)
        elif df['age'].iloc[i]>39:
            age_group_3.append(i)
            
    
    
    Xtr1 = np.empty((0,0))
    Ytr1 = np.empty(0)
    for i in age_group_1:
        if np.size(Xtr1)==0:
            print("bismillah")
            Xtr1 = X[i]
            Ytr1 = y[i]
        else:
            Xtr1 = np.vstack((Xtr1,X[i]))
            Ytr1 = np.append(Ytr1,y[i])

    Xtr2 = np.empty((0,0))
    Ytr2 = np.empty(0)
    for i in age_group_2:
        if np.size(Xtr2)==0:
            print("bismillah")
            Xtr2 = X[i]
            Ytr2 = y[i]
        else:
            Xtr2 = np.vstack((Xtr2,X[i]))
            Ytr2 = np.append(Ytr2,y[i])
        
    Xtr3 = np.empty((0,0))
    Ytr3 = np.empty(0)
    for i in age_group_3:
        if np.size(Xtr3)==0:
            print("bismillah")
            Xtr3 = X[i]
            Ytr3 = y[i]
        else:
            Xtr3 = np.vstack((Xtr3,X[i]))
            Ytr3 = np.append(Ytr3,y[i])
    
    clients = {}
    client_data_testx = []
    client_data_testy = []
    x_train, x_test, y_train, y_test = train_test_split(Xtr1,Ytr1,test_size=0.2)
    Xtr1 = x_train
    Xte1 = x_test
    Ytr1 = y_train
    Yte1 = y_test
    Xtr = x_train
    client_data_testx.append(Xte1)
    client_data_testy.append(Yte1)
    ####
    x_train, x_test, y_train, y_test = train_test_split(Xtr2,Ytr2,test_size=0.2)
    Xtr2 = x_train
    Xte2 = x_test
    Ytr2 = y_train
    Yte2 = y_test
    client_data_testx.append(Xte2)
    client_data_testy.append(Yte2)
    ####
    x_train, x_test, y_train, y_test = train_test_split(Xtr3,Ytr3,test_size=0.2)
    Xtr3 = x_train
    Xte3 = x_test
    Ytr3 = y_train
    Yte3 = y_test
    client_data_testx.append(Xte3)
    client_data_testy.append(Yte3)
    
    #concatnate teset data
    x_test_new = np.concatenate((client_data_testx[0], client_data_testx[1]), axis=0)
    x_test_new = np.concatenate((x_test_new, client_data_testx[2]), axis=0)
    y_test_new = np.concatenate((client_data_testy[0], client_data_testy[1]), axis=0)
    y_test_new = np.concatenate((y_test_new, client_data_testy[2]), axis=0)
    #test_batched1 = tf.data.Dataset.from_tensor_slices((x_test_new, y_test_new)).batch(len(y_test_new))
    x_test = x_test_new
    y_test = y_test_new
    
    labels = Ytr3
    unique, counts = np.unique(labels, return_counts=True)
    count_ap_dict = dict(zip(unique, counts))
    client_index = {}
    client_window = {}
    client_window_label = {}
    client_eddm = {}
    clients = create_clients_attr(Xtr1,Ytr1,Xtr2,Ytr2,Xtr3,Ytr3, num_clients=3, initial='client')
    for (client_name, data) in clients.items():
        data, label = zip(*data)
        Y = np.asarray(label)
        X = np.asarray(data)
        client_index.update({client_name:0})
        client_window.update({client_name:[]})
        client_window_label.update({client_name:[]})
        length = len(data)

    
    return clients, client_index, client_window, client_window_label, client_eddm, length, p_Group, np_Group, sa_index


    