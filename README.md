# Robust Graph Clustering via Meta Learning for Noisy Graphs

MetaGC is a GNN-based graph clustering method that is robust against noise edges. 
MetaGC consists of a GNN-based clustering model, and a meta model that adaptively adjusts
the weights of node pairs.
MetaGC uses a modularity-based loss function with theoretical justification.
In our experiments, we demonstrate that MetaGC learns weights effectively and 
thus outperforms the state-of-the-art GNN-based competitors, even when 
they are equipped with separate denoising schemes, on five real-world graphs under varying levels of noise. 

## Environments
- python 3.7.11
- numpy==1.21.2
- torch==1.10.0 (with CUDA 11.3)
- sklearn==1.0.2
- scipy==1.7.3

## important argments
- noise_level: noise level, [1, 2, 3] (1 --> 30%, 2 --> 60%, 3 --> 90%)
- batch_size: set the batch sizes (i.e., the numbers of nodes in a batch)
- nuim_hiddens: number of hidden units in the GCN variant
- num_epochs: minimum epochs
- max_epochs: maximum epochs
- num_pateince: number of epochs with no improvement of modularity after which training will be stopped
- c_lr: learning rate of the clustering model
- m_lr: learning rate of the meta model

## Running MetaGC with Cora graph with noise level I
python3 main.py --graph_name cora --noise_level 1 --batch_size 128 --num_hiddens 64 --num_epochs 200 --max_epochs 1500 --num_patience 50 --c_lr 0.001 --m_lr 0.005


