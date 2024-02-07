# LA-SCL
Repo for [Learning label hierarchy with supervised contrastive learning](https://arxiv.org/abs/2402.00232)

Training:
```
bash run.sh
```
Training Tutorials:
Substitute flags in run.sh as shown below for different methods.
```
supcl: Original supervised contrastive
--method supcl

LI: Label-aware instance-instance contrastive
--method label_string
--update_label_emb

LIUC: Label-aware instance-to-unweighted-center
--method supcl
--update_label_emb
--kmeans

LIC: Label-aware instance-to-center
--method label_string
--update_label_emb
--kmeans

LISC: Label-aware instance-to-scaled-center
--method label_string
--update_label_emb
--modified_kmeans
```

Evaluation:
```
bash evaluation/run.sh
```
Evaluation Tutorials:
```
--train_file corresponds to training set for linear probing
--valid_file corresponds to validation set for linear probing
--dataset sepcify dataset name corresponds to huggingface's dataset names, will automatically download test set (except for wos, which is explained below)
--task direct_test/lp/lp_random_initialized_linear/finetune 
```

Explanation to data folder
* Linear Probing: newsgroups20, dbpedia, wos (wos contains one additional test.csv)
* WebOfScience: Original WOS dataset used for training
* label_str.pkl pre-generated WOS label mappings