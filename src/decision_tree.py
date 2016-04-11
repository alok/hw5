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

    class Node(params):

        """Docstring for Node. """

        def __init__(self):
            """TODO: to be defined1. """
            params.__init__(self)

        def segmenter(data, labels):

    def impurity(left_label_hist, right_label_hist):


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

