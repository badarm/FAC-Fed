# FAC-Fed: Federated adaptation for fairness and concept drift aware stream classification
Federated Learning (FL) is an emerging collaborative learning paradigm of Machine Learning (ML) involving distributed and heterogeneous clients. Enormous collections of continuously arriving heterogeneous data residing on distributed clients require federated adaptation of efficient mining algorithms to enable fair and high-quality predictions with privacy guarantees and minimal response delay. In this context, we propose a federated adaptation that mitigates discrimination embedded in the streaming data while handling concept drifts (FAC-Fed). We present a novel adaptive data augmentation method that mitigates client-side discrimination embedded in the data during optimization, resulting in an optimized and fair centralized server. Extensive experiments on a set of publicly available streaming and static datasets confirm the effectiveness of the proposed method. To the best of our knowledge, this work is the first attempt towards fairness-aware federated adaptation for stream classification, therefore, to prove the superiority of our proposed method over state-of-the-art, we compare the centralized version of our proposed method with three centralized stream classification baseline models (FABBOO, FAHT, CSMOTE). The experimental results show that our method outperforms the current methods in terms of both discrimination mitigation and predictive performance.
## The datsets used in this project
* [Adult Census](https://archive.ics.uci.edu/dataset/2/adult)
* [Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
* [Default](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
* [Law School](https://github.com/iosifidisvasileios/FABBOO/blob/master/Data/law_dataset.arff)
## Code
### Dataset Processing Scripts

The `datasets` directory contains all the datasets used in this project. Below is a description of python scripts written to process each dataset:

- `load_adult.py`: Utility script for loading and preprocessing the 'Adult' dataset commonly used in machine learning tasks related to income prediction.
  
- `load_bank.py`: This script is designed to load and preprocess the 'Bank' dataset, which includes data for marketing campaigns of banking institutions.

- `load_default.py`: A script that handles the loading and initial processing of the 'Default' dataset, which might be related to credit card defaulting data (Replace with the actual functionality).

- `load_law.py`: Loads the 'Law' dataset and prepares it for analysis, possibly containing information about legal precedents or case outcomes (Replace with the actual functionality).
### FAC-Fed main scripts
The following scripts constitute the complete methodology of FAC-Fed
- `facfed_main.py`: Main script for the 'FacFed' framework that orchestrates the fairness aware federated learning process on different datasets.
- `cfsote.py`: The script contains functions related to the 'CFSOTE' algorithm for adaptive data augmnetation for discrimination mitigation.
  
- `onn.py`: Contains the implementation of an Online Neural Network (ONN) used as a bse model in FAC-Fed.

## Running the facfed_main.py Script

To run the `facfed_main.py` script with the default settings, you can use the following command:

```bash
python facfed_main.py --num_clients 3 --fairness_notion 'eqop' --dataset_name 'bank' --distribution 'attr'


## Prerequisities
* numpy
* pandas
* sklearn
* scipy
* matplotlib
* tensorflow
* pytorch
* scikit-multiflow


