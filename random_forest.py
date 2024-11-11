import numpy as np
from collections import Counter

# Helper function: Calculate Gini Impurity
def gini_impurity(y):
    proportions = np.bincount(y) / len(y)
    return 1 - sum(p ** 2 for p in proportions)

# Helper function: Split dataset based on feature and threshold
def split_data(X, y, feature, threshold):
    left_idx = np.where(X[:, feature] <= threshold)[0]
    right_idx = np.where(X[:, feature] > threshold)[0]
    return X[left_idx], y[left_idx], X[right_idx], y[right_idx]

# Decision Node
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

# Decision Tree
class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            return Node(value=self._most_common_label(y))

        # Find the best split
        best_feature, best_threshold = self._best_split(X, y, n_features)
        if best_feature is None:
            return Node(value=self._most_common_label(y))

        # Recursive tree building
        left_X, left_y, right_X, right_y = split_data(X, y, best_feature, best_threshold)
        left = self._grow_tree(left_X, left_y, depth + 1)
        right = self._grow_tree(right_X, right_y, depth + 1)
        return Node(best_feature, best_threshold, left, right)

    def _best_split(self, X, y, n_features):
        best_gini = 1
        best_feature, best_threshold = None, None
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                _, left_y, _, right_y = split_data(X, y, feature, threshold)
                if len(left_y) == 0 or len(right_y) == 0:
                    continue

                gini = (len(left_y) * gini_impurity(left_y) + len(right_y) * gini_impurity(right_y)) / len(y)
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold

    def _most_common_label(self, y):
        return Counter(y).most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

# Random Forest
class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return [Counter(tree_preds[:, i]).most_common(1)[0][0] for i in range(X.shape[0])]

# Test the implementation
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Load dataset
    data = load_iris()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train Random Forest
    rf = RandomForest(n_trees=10, max_depth=10, min_samples_split=2)
    rf.fit(X_train, y_train)

    # Make predictions
    y_pred = rf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
