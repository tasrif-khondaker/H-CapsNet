# H-CapsNet: A Capsule Network for Hierarchical Image Classification #

## highlights ##
* We propose a new CapsNet for hierarchical classification containing a dedicated capsule network for each hierarchical level. In this manner, our network can deliver multiple class predictions based on a hierarchical label-tree.
* We enforce consistency with the label-tree by making use of a modified hinge-loss that takes into account both, the number of classes within each hierarchy and the relationship between each of these in the label tree. 
*  We present a strategy to dynamically adjust the training parameters in order to moderate the contribution of each hierarchical level to the loss. This is shown to improve performance while balancing the contribution of each of the hierarchical levels making use of the classification error. 
* We show results on widely available datasets (MNIST, EMNIST, Fashion-MNIST, Marine-tree, CIFAR-10 and CIFAR-100) and compare them against alternatives elsewhere in the literature. Our experiments show that H-CapsNet delivers a margin of advantage over the alternatives and converges faster.

# CITATION
To acknowledge the use of the code for H-CapsNet model in publications, please cite the following paper:
- Noor, Khondaker Tasrif, and Antonio Robles-Kelly. "H-Capsnet: A Capsule Network for Hierarchical Image Classification." Available at SSRN 4271318.

# HOW TO TRAIN
For training the H-CapsNet model please use "H-CapsNet_Training.ipynb" file. The default parameter settings is given in the notebook.

For obtaining better results please use trained model weights (provided in the corresponding directories) for each directory.

* Note: To use Marine-tree dataset additional download is required. For detail instruction on how to download marine-tree dataset visit following link:
https://github.com/tboone91/Marine-tree

# HOW TO ANALYSE
To analyse H-CapsNet model use the "H-CapsNet-Analysis.ipynb" file. The file required pre-trained model weights to evaluate on each dataset. Please use the provided weights or use the "H-CapsNet_Training.ipynb" file to train and generate model weights before evaluate the models.