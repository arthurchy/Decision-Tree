from collections import Counter
import math
import random
import numpy as np

import DecisionTreeNode

# Pre-pruning max tree depth, criterion
class DecisionTree:
    """
    Pre-pruning max tree depth, criterion
    tuning parameters: (the definition is from sklearn DecisionTreeClassifier)
    https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

    Attributes:
        criterion: gini or entropy impurity , default gini
        max_depth:  maximum depth of the decision tree, default None
        min_samples_leaf: minimum number of sample in leaf node, default 1
        min_samples_split: minimum number of samples required to split an internal node, default 2
        max_features: number of features select for split, default none
            if intput = int, features = input
            if input = float, features = max(1, int(max_features * n_feature)
            if input = "sqrt", features = sqrt(n_feature)
            if input = "log2", features = log2(n_feature)
    """

    def __init__(self, criterion = "gini", max_depth = None, min_samples_leaf = 1, min_samples_split = 2, max_features = None, random_state = None):
        """hyperparameters"""
        self.tree = None
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state

        self.labels_classes = None # a list contain the class labels, use to identify muliclass problem where labels not lying on [0,1]
        self.total_feature = None # checking the number of feature matches training set

        if random_state is not None:
            random.seed(random_state)

    
    def _validate_params(self):
        """function to validate input parameters are valid input"""
        # check if criterion is either gini or entropy
        if self.criterion not in ("gini", "entropy"):
            raise ValueError("Criterion must be gini or entropy")
        
        # check max_depth is none or int
        if self.max_depth is not None and (not isinstance(self.max_depth, int) or self.max_depth < 1):
            raise ValueError("max_depth must be none or integer >= 1")

        # check min_samples_leaf is int and >= 1
        if not isinstance(self.min_samples_leaf, int) or self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be integer >= 1")

        # check min_samples_split is int and >= 2
        if not isinstance(self.min_samples_split, int) or self.min_samples_split < 2:
            raise ValueError("min_samples_split must be integer >= 2")
        
        # check if max_features is None, sqrt, log2, int or float
        if self.max_features is not None:
            if isinstance(self.max_features, str) and self.max_features not in ("sqrt", "log2"):
                raise ValueError("max_features must be an int, float, sqrt, log2 or None")
            if (isinstance(self.max_features, int) or isinstance(self.max_features, float)) and self.max_features <= 0:
                raise ValueError("max_features must be > 0 ")


    def _calculate_impurity(self, data_count: Counter, total_sample: int) -> float:
        """function to calculate impuirty (gini or entropy)"""
        if total_sample == 0:
            return 0
        
        probability = np.array(list(data_count.values())) / total_sample

        if self.criterion == "gini":
            return 1.0 - np.sum(probability ** 2)
        elif self.criterion == "entropy":
            probability = probability[probability > 0]      # ignore probability = 0, prevent log2(0)
            return -np.sum(probability * np.log2(probability))
    
    def _find_best_split(self, data, features):
        """
        function to find the best split

        logic: for each feature, find the best split
        
        """
        labels_count = Counter()
        for row in data:
            label = row[-1]
            labels_count[label] += 1
        best_split = {
            "score": float("inf"), 
            "feature": None, 
            "split_value": None, 
            "left": None, 
            "right": None,
            "current_impurity": None}
        best_split_list = []
        min_score_list = []
        total_samples = len(data)
        current_score = self._calculate_impurity(labels_count, total_samples)

        for feature in features:
            feature_values = data[:, feature]
            sorted_indices = np.argsort(feature_values)
            sorted_data = data[sorted_indices]

            left_counts = Counter()
            right_counts = Counter(sorted_data[:, -1])
            left_size, right_size = 0, total_samples

            for i in range(1, total_samples):
                label = sorted_data[i - 1, -1]  # Label of the current sample
                left_counts[label] += 1
                if right_counts[label] > 0:
                    right_counts[label] -= 1
                left_size += 1
                right_size -= 1

                # skip if the feature value are same in sorted list
                if feature_values[sorted_indices[i]] == feature_values[sorted_indices[i - 1]]:
                    continue
                
                # skip when the size is similar than required min_samples_leaf
                if left_size < self.min_samples_leaf or right_size <self.min_samples_leaf:
                    continue

                left_score = self._calculate_impurity(left_counts, left_size) 
                right_score = self._calculate_impurity(right_counts, right_size)

                total_score = (left_size / total_samples) * left_score + (right_size / total_samples) * right_score
                if total_score <= best_split["score"]:
                    best_split_list.append({"score": total_score, "feature": feature, "split_value": (feature_values[sorted_indices[i]] + feature_values[sorted_indices[i - 1]]) / 2, "left": sorted_data[:i], "right": sorted_data[i:], "current_impurity": current_score})
                    best_split.update({"score": total_score, "feature": feature, "split_value": (feature_values[sorted_indices[i]] + feature_values[sorted_indices[i - 1]]) / 2, "left": sorted_data[:i], "right": sorted_data[i:], "current_impurity": current_score})

        if len(best_split_list) == 0:   # no split found, return empty best_split dict
            return best_split

        # add all best split into the list
        for split in best_split_list:
            if split["score"] == best_split["score"]:
                min_score_list.append(split)

        # random pick one if more than one best split found
        return random.choice(min_score_list)

    # private function to check the maximum feature
    # the function would check the input of maximum feature
    def _max_feature_check(self, n_features):
        # if self.max_features = None, return the full list of feature as candidate 
        if self.max_features is None:
            return list(range(n_features))
        
        # if input is int, then random sample the number of feature equal to input
        if isinstance(self.max_features, int):
            return random.sample(range(n_features), self.max_features)
        
        # if input is float, then random sample the number of feature equal to the max of (input * number of feature or 1)
        elif isinstance(self.max_features, float):
            return random.sample(range(n_features), max(1, int(self.max_features * n_features))) 
        
        # if input is sqrtm, then random sample the number of feature equal to square root of number of feature
        elif self.max_features == "sqrt":
            return random.sample(range(n_features),  max(1, int(math.sqrt(n_features))))
            # if input is sqrtm, then random sample the number of feature equal to log2 of number of feature
        
        elif self.max_features == "log2":
            return random.sample(range(n_features),  max(1, int(math.log2(n_features))))
        
        return list(range(n_features))

    def _build_tree(self, data, depth = 0):
        labels_count = Counter()

        for row in data:
            label = row[-1]
            labels_count[label] += 1
        n_sample = len(data)

        # checking current depth > max depth
        if self.max_depth is not None and depth >= self.max_depth:
            return DecisionTreeNode.DecisionTreeNode(labels = dict(labels_count), samples = n_sample)
        
        # checking current number of sample < min_samples_leaf
        if self.min_samples_leaf > n_sample:
            return DecisionTreeNode.DecisionTreeNode(labels = dict(labels_count), samples = n_sample)
        
        # if there are only 1 label with count > 0, no need to split
        label_count_greater_zero = sum(1 for label in labels_count.values() if label > 0)
        if label_count_greater_zero == 1:
            return DecisionTreeNode.DecisionTreeNode(labels = dict(labels_count), samples = n_sample)

        # receive the resulting feature list from checking function
        feature_list = self._max_feature_check(len(data[0]) - 1)
        best_split = self._find_best_split(data, feature_list)

        if best_split["score"] == float("inf"):
            # No valid split found, create a leaf node
            return DecisionTreeNode.DecisionTreeNode(labels=dict(labels_count), samples = n_sample)

        left_tree = self._build_tree(best_split["left"], depth + 1)
        right_tree = self._build_tree(best_split["right"], depth + 1)
        
        return DecisionTreeNode.DecisionTreeNode(
            split_value = best_split["split_value"],
            feature = best_split["feature"],
            score = best_split["current_impurity"],
            left = left_tree,
            right = right_tree,
            labels = None,  # Internal nodes don't need label counts
            samples = n_sample
        )

                
    # fit function
    # added new function including getting the label list, and pass handle minimum split after tree is built
    def fit(self, data, labels):
        self.total_feature = len(data[0])
        # datas = np.array([row + [label] for row, label in zip(data, labels)])  # Combine data and labels
        datas = np.hstack([np.array(data), np.array(labels).reshape(-1, 1)])
        self.tree = self._build_tree(datas)
        self.labels_classes = sorted(set(labels))
        return self

    def _go_tree(self, data_point):
        current = self.tree
        while current.left is not None and current.right is not None:
            if data_point[current.feature] <= current.split_value:
                current = current.left
            else:
                current = current.right

        max_count = max(current.labels.values())

        label_list = []
        for label, label_count in current.labels.items():
            if label_count == max_count:
                label_list.append(label)
        
        if len(label_list) > 1:
            return random.choice(label_list)
        else:
            return label_list[0]

    def predict(self, data):
        n_feature = len(data[0])
        if self.total_feature != n_feature:
            raise ValueError("Number of feature does not match with trained data")
        result_list = []
        for row in data:
            result_list.append(self._go_tree(row))
        return result_list




    







