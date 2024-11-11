from collections import Counter

import numpy as np


class DecisionTree:
    def __init__(self, min_samples_split = 2, max_depth=1000, n_features = None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        y = np.array(y)
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        # we will make the tree by calling the make tree
        self.root = self._make_tree(X, y)

    def predict(self, X):
        return np.array([self._tree_traverse(x, self.root) for x in X])

    def _tree_traverse(self, x, node):
        if node.is_leaf():
            return node.value
        else:
            feature = node.feature
            threshold = node.threshold
            if x[feature] <= threshold:
                return self._tree_traverse(x, node.left)
            else:
                return self._tree_traverse(x, node.right)


    def _make_tree(self, X, y, depth = 0):
        #getting the sample size/row and feature/col from X
        n_samples, n_true_feature = X.shape
        #getting the len of unique y value to get the label as the leaf node only have one label.
        n_labels = len(np.unique(y))

        #check the stopping criteria (if it is leaves node n_labels, or the depth is greater than our set value or
        # sample number(row number) is less than our define sample number.
        if n_labels == 1 or depth >= self.max_depth or n_samples < self.min_samples_split:
            # calculate the value of the node
            #if there is only one label then we will just get the y value of that label and return
            # However, if there are more than one node, we will do the majority vote
            return Node(Counter(y).most_common(1)[0][0])
        #find the best split position
        feature_indexes = np.random.choice(n_true_feature, self.n_features, replace=False)
        best_feature, best_threshold = self._best_split(X, y, feature_indexes)
        #create child nodes

        left_child, right_child = self._split(X[:, best_feature], best_threshold)
        left = self._make_tree(X[left_child,:], y[left_child], depth + 1)
        right = self._make_tree(X[right_child,:], y[right_child], depth + 1)
        return Node(best_feature, best_threshold, left, right)

    def _best_split(self, X, y, feature_indexes):
        # based on the information gain and entropy function, we will decide which feature (attribute) is best for splitting.
        # Threshold -> a value within a selected feature that divide the data into two node.

        max_info_gain = -1
        best_feature, best_threshold = None, None
        for feature_index in feature_indexes:
            #taking a particular columns and getting the unique value
            x_feature = X[:, feature_index]
            thresholds = np.unique(x_feature)
            for threshold in thresholds:
                # calculate information gain for each value
                info_gain = self._cal_info_gain(x_feature, y, threshold)
                #update our best feature and threshold value
                if info_gain > max_info_gain:
                    max_info_gain = info_gain
                    best_feature = feature_index
                    best_threshold = threshold
        return best_feature, best_threshold


    def _cal_info_gain(self, x, y, threshold):
        #formulla for info gain = entropy(parent) - wighted average * entropy(child)
        parent_entropy = self._entropy(y)
        #create children based on the threshold I got
        left_child_index, right_child_index = self._split(x, threshold)
        if len(left_child_index) == 0 or len(right_child_index) == 0:
            return 0
        #calculcated the weighted average * entropy(child)
        N = len(y)
        left_weight = len(left_child_index) / N
        right_weight = len(right_child_index) / N

        # Ensure y is correctly indexed
        left_entropy = self._entropy(y[left_child_index])
        right_entropy = self._entropy(y[right_child_index])
        child_entropy = left_weight * left_entropy + right_weight * right_entropy

        # left_entropy = self._entropy(y[left_child_index])
        # right_entropy = self._entropy(y[right_child_index])
        # child_entropy = left_weight * left_entropy + right_weight * right_entropy
        #calculcate the info gain
        info_gain = parent_entropy - child_entropy
        return info_gain

    def _entropy(self, y):
        #fromuala for entropy is E = - SUM ( p(x) * log2(P(x))
        x = np.bincount(y) # counter the number of occurrence in y
        ps = x / len(y) #probability
        entropy = -np.sum([p * np.log2(p) for p in ps if p > 0])
        return entropy

    def _split(self,x_feature,threshold):
        left_child_index = np.argwhere(x_feature <= threshold).flatten()
        right_child_index = np.argwhere(x_feature > threshold).flatten()
        return left_child_index, right_child_index

#Every interval node will have a feature and threshold based those feature, threshold we will decide which way should we have to go
#Only leaf node should value.
class Node:
    def __init__(self, feature=None, threshold=None, left = None, right = None, *, value = None):
        self.left = None
        self.right = None
        self.feature = None
        self.threshold = None
        self.value = value

    def is_leaf(self):
        # it will check whether the ndoe is leaf node or not
        return self.value is not None
