# LA-SCL
Repo for [Learning label hierarchy with supervised contrastive learning](https://arxiv.org/abs/2402.00232)

Training:
```
bash run.sh
```
Training Tutorials:
```
*supcl: Original supervised contrastive
--method supcl

*LI: Label-aware instance-instance contrastive
--method label_string
--update_label_emb

*LIUC: Label-aware instance-to-unweighted-center
--method: supcl
--update_label_emb
--kmeans

*LIC: Label-aware instance-to-center
--method: label_string
--update_label_emb
--kmeans

*LISC: Label-aware instance-to-scaled-center
--method: label_string
--update_label_emb
--modified_kmeans
```

