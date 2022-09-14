# H-CapsNet: A Capsule Network for Hierarchical Image Classification #

## highlights ##
* We propose a new CapsNet for hierarchical classification containing a dedicated capsule network for each hierarchical level. In this manner, our network can deliver multiple class predictions based on a hierarchical label-tree.
* We enforce consistency with the label-tree by making use of a modified hinge-loss that takes into account both, the number of classes within each hierarchy and the relationship between each of these in the label tree. 
*  We present a strategy to dynamically adjust the training parameters in order to moderate the contribution of each hierarchical level to the loss. This is shown to improve performance while balancing the contribution of each of the hierarchical levels making use of the classification error. 
* We show results on widely available datasets (EMNIST, Fashion-MNIST, CIFAR-10 and CIFAR-100) and compare them against alternatives elsewhere in the literature. Our experiments show that H-CapsNet delivers a margin of advantage over the alternatives and converges faster.
