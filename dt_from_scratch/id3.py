from typing import Tuple
import numpy as np


class Node:
    def __init__(
        self,
        feature=None,
        threshold=None,
        left=None,
        right=None,
        value=None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class ID3:
    def __init__(self, max_depth: int = None) -> None:
        self.max_depth = max_depth
        self.n_features = None
        self.tree = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the decision tree classifier (ID3)
        """
        self.n_features = X.shape[1]
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """
        Recursively grow the decision tree

        Parameters:
            X (np.ndarray): The input data
            y (np.ndarray): The target labels
            depth: The current depth of the tree (stop growing when depth >= max_depth)

        Returns:
            Node: The root node of the decision tree
        """
        n_labels = len(np.unique(y))  # number of unique classes

        if (self.max_depth is not None and depth >= self.max_depth) or n_labels == 1:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        best_feature, best_threshold = self._best_split(X, y)

        left_idxs, right_idxs = X[:, best_feature] < best_threshold, X[:, best_feature] >= best_threshold

        # Recursively grow the left and right subtree
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)

        return Node(best_feature, best_threshold, left, right)

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
        """
        Split criteria of ID3 algorithm: Information Gain

        Parameters:
            X (np.ndarray): The input data
            y (np.ndarray): The target labels

        Returns:
            Tuple[int, float]: The best feature index and threshold
        """
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature in range(self.n_features):  # iterate over all features
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:  # iterate over all unique values of the feature
                gain = self._information_gain(X[:, feature], y, threshold)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, X_column: np.ndarray, y: np.ndarray, threshold: float) -> float:
        """
        Calculate the information gain

        Parameters:
            X_column (np.ndarray): The feature column
            y (np.ndarray): The target labels
            threshold (float): The threshold value

        Returns:
            float: The information gain

        Math equation:
            $IG(D, A) = H(D) - H(D|A)$

            where D is the dataset and A is the feature
        """
        parent_entropy = self._entropy(y)

        left_idxs, right_idxs = X_column < threshold, X_column >= threshold

        if len(left_idxs) == 0 or len(right_idxs) == 0:  # split is not possible
            return 0

        # Calculate the weighted avg. of the entropy for the children
        n = len(y)
        n_l, n_r = len(y[left_idxs]), len(y[right_idxs])
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        information_gain = parent_entropy - child_entropy  # the difference before and after the split
        return information_gain

    def _entropy(self, y: np.ndarray) -> float:
        """
        The entropy of the dataset

        bincount: np.bincount([0, 1, 2, 1, 3, 5, 2, 3]) = [1, 2, 2, 2, 0, 1]

        Entropy formula:
            $H(D) = -\sum_{i=1}^n p_i \log_2(p_i)$
        """
        percentage = np.bincount(y) / len(y)  # percentage of each class
        return -np.sum([p * np.log2(p) for p in percentage if p > 0])

    def _most_common_label(self, y: np.ndarray) -> int:
        """
        Get the most common label in the target labels
        """
        return np.argmax(np.bincount(y))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the target labels using testing dataset
        """
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x: np.ndarray, node: Node) -> int:
        if node.value is not None:
            return node.value

        if x[node.feature] < node.threshold:
            return self._traverse_tree(x, node.left)

        return self._traverse_tree(x, node.right)
