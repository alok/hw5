#!/usr/bin/env python3
# encoding: utf-8

class DecisionTree(params):

    """A binary tree that takes in

    : split rule : (feat to split on (int index of data pt's feat vector), threshold to split on)
    :left: left child of current node
    :right: right child of current node
    :label: if set, then Node is a leaf and it contains the desired label
    """

    def __init__(self):
        """TODO: to be defined1. """
        params.__init__(self)

    class Node(split_rule, left, right, label):

        def __init__(self):

# split_rule: A length 2 tuple that details what feature to split on at a node, as well as the threshold value at which you should split at. The former can be encoded as an integer index into your data point’s feature vector.
# left: The left child of the current node.
# right: The left child of the current node.
# label If this field is set, the Node is a leaf node, and the field contains the label with which you should classify a data point as, assuming you reached this node during your classification tree traversal. Typically, the label is the mode of the labels of the training data points arriving at this node.


# impurity(left_label_hist, right_label_hist): A method that takes in the result of a split: two histograms (a histogram is a mapping from label values to their frequencies) that count the frequencies of labels on the ”left” and ”right” side of that split. The method calculates and outputs a scalar value representing the impurity (i.e. the ”badness”) of the specified split on the input data.

# segmenter(data, labels): A method that takes in data and labels. When called, it finds the best split rule for a Node using the impurity measure and input data.
# There are many different types of segmenters you might implement, each with a different method of choosing a threshold.
# The usual method is exhaustively trying lots of different threshold values from the data and choosing the combination of split feature and threshold with the lowest impurity value.
# The final split rule uses the split feature with the lowest impurity value and the threshold chosen by the segmenter.
# Be careful how you implement this method! Your classifier might train very slowly if you implement this badly.

    def impurity(left_label_hist, right_label_hist):

    def segmenter(data = data, labels = labels):


    def train(data, labels):
        """
        - Grows a decision tree by constructing nodes.
        - Using the impurity and segmenter methods, attempts to find a configuration of nodes that best splits the input data.
        - Figures out the split rules that each node should have and figures out
        when to stop growing the tree and insert a leaf node.
        - Your DecisionTree should store the root node of the resulting tree so you can use the tree for
        classification later on.
         - The height of your DecisionTree shouldn’t be astronomically large (you may want to cap the height - if
        you do, the max height would be a hyperparameter), this method is best
        implemented recursively.
        """

    def predict(data):
        """
        Given a data point, traverse the tree to find the best label to
        classify the data point as. Start at the root node you stored and
        evaluate split rules at each node as you traverse until you reach a
        leaf node, then choose that leaf node’s label as your output label.
        """

def H(S = data, y = labels):
    """
    entropy of a set
    """

    p_0 = len( [i for i in range(len(data)) if y[i] == 0]) / len(S)
    p_1 = len( [i for i in range(len(data)) if y[i] == 1]) / len(S)
    p = [p_0, p_1]
    h = p
    for i, pr in enumerate(p):
        if pr == 0:
            h[i] = 0
        else:
            h[i] = pr * math.log(pr, 2)

    return -np.sum(h)
