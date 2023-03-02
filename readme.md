# Transformer-based Hierarchical Clustering

This repository contains the code for the paper "Transformer-Based Hierarchical Clustering for Brain Network Analysis," to be published at IEEE-ISBI 2023. 

## Dataset

ABIDE can be accessed from [here](https://fcon_1000.projects.nitrc.org/indi/abide/). The preprocessing scripts can be found at util/abide, with instructions in the same folder. 

## Usage


### ABIDE

```bash
python main.py --config_filename setting/abide_dec_ortho_learnable_newpool_sftmx_topk_90_4_hierarchical_sum_5-4lr.yaml
```

### ABCD

```bash
python main.py --config_filename setting/abcd_dec_ortho_learnable_newpool_sftmx_topk_90_4_hierarchical_sum.yaml
```