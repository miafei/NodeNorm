# NodeNorm - Understanding and Resolving Performance Degradation in Deep Convolutional Graph Neural Networks
Official pytorch code for NodeNorm [paper](https://arxiv.org/pdf/2006.07107.pdf) (CIKM 2021)  

## Eligible Version of Python and Related Packages

| Item | Version|
| ---- | ---- |
| Python | 3.7.5 |
| DGL | 0.4.1 |
| Pytorch_Geometric| 1.4.3 |
| Pytorch | 1.4.0 |
| Networkx | 2.3 |
| Sacred | 0.8.1 |
| Cuda | 9.2 or 10.1|

## Run with Sacred

### run directly

`python main.py`

### run with automatic debugging mode

`python main.py -D`


### run with manual debugging mode

`python main.py -d`

### run with specified options

We use "with" to denote modified options in Sacred config.
Here the key of a dictionary can be used as a property of that "dictionary".
Options not specified after "with" will remain the default values in config.py.

For example, to reproduce the result of 2-layer GCN-res with NodeNorm, run

```python
python main.py with data.dataset=cora arch.layer_type=gcn_res arch.block_type=n_a_r arch.num_layers=2 arch.dropout.p=0.8 optim.l1_weight=0.001 optim.weight_decay=0.001
```

In the command above, 'arch.layer_type' can be chosed from:
<br>'gcn' for GCN
<br>'gcn_res' for GCN with residual connections
<br>'gat' for GAT
<br>'gat_res' for GAT with residual connections
<br>'sage' for GraphSage
<br>'sage_res' for GraphSage with residual connections;

'arch.block_type' can be chosed from:
<br>'v' for vallina layers without residual connection or normalization
<br>'a_r' for adding residual connections
<br>'b_a' for adding BatchNorm
<br>'n_a' for adding NodeNorm
<br>'b_a_r' for adding both BatchNorm and residual connections
<br>'n_a_r' for adding both NodeNorm and residual connections.

By setteing 'data.random_split.use=False', we can use the commonly used split for Cora, Citeseer and Pubmed as used in [1].

We may set 'arch.nn=False' to switch off NodeNorm for the first layer.

In our code, we use the DGL library for building our GNN architectures and the Pytorch_Geometric library for providing data.


## cite 
If you use our code, please cite
```
@inproceedings{zhou2021understanding,
  title={Understanding and Resolving Performance Degradation in Deep Graph Convolutional Networks},
  author={Zhou, Kuangqi and Dong, Yanfei and Wang, Kaixin and Lee, Wee Sun and Hooi, Bryan and Xu, Huan and Feng, Jiashi},
  booktitle={Proceedings of the 30th ACM International Conference on Information \& Knowledge Management},
  pages={2728--2737},
  year={2021}
}
```


***

**Reference**
<br>[1] Kipf, T. N., & Welling, M. (2016). Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907.
