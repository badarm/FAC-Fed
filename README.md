# FAC-Fed: Federated adaptation for fairness and concept drift aware stream classification
Federated Learning (FL) is an emerging collaborative learning paradigm of Machine Learning (ML) involving distributed and heterogeneous clients. Enormous collections of continuously arriving heterogeneous data residing on distributed clients require federated adaptation of efficient mining algorithms to enable fair and high-quality predictions with privacy guarantees and minimal response delay. In this context, we propose a federated adaptation that mitigates discrimination embedded in the streaming data while handling concept drifts (FAC-Fed). We present a novel adaptive data augmentation method that mitigates client-side discrimination embedded in the data during optimization, resulting in an optimized and fair centralized server. Extensive experiments on a set of publicly available streaming and static datasets confirm the effectiveness of the proposed method. To the best of our knowledge, this work is the first attempt towards fairness-aware federated adaptation for stream classification, therefore, to prove the superiority of our proposed method over state-of-the-art, we compare the centralized version of our proposed method with three centralized stream classification baseline models (FABBOO, FAHT, CSMOTE). The experimental results show that our method outperforms the current methods in terms of both discrimination mitigation and predictive performance.
## The datsets used in this project
* [Adult Census](https://archive.ics.uci.edu/dataset/2/adult)
* Bank Marketing
* Default
* Law School
