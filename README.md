# Group Equivariant Capsule Networks in PyTorch
Proof of concept Pytorch implementation of [Group Equivariant Capsule Networks](https://arxiv.org/pdf/1806.05086.pdf) (NIPS 2018).

#### Required:
[PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric)

#### Install:
1) Clone Repo
2) `python setup.py install`

In repo dir: `python examples/mnist.py` to run the MNIST Experiment.

Implementations of capsules in [pooling_capsule_layer.py](https://github.com/mrjel/group_equivariant_capsules_pytorch/blob/master/group_capsules/nn/modules/pooling_capsule_layer.py)

Implementations of sparse group convolution in [group_conv_layer.py](https://github.com/mrjel/group_equivariant_capsules_pytorch/blob/master/group_capsules/nn/modules/group_conv_layer.py)
