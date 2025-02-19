class DecisionTreeNode:
    """
    create class for each tree node

    Parameters:
        split_value: middle value for splitting into 2 branches
        left: left subtree
        right: right subtree
        feature: index of the feature used for split
        labels: counts of each class label 
        samples: number of sample at the node
    
    """
    def __init__(self, split_value = None, left = None, right = None, feature = None, score = None, labels = None, samples = None):
        self.split_value = split_value
        self.left = left
        self.right = right
        self.feature = feature
        self.score = score
        self.labels = labels or {}
        self.samples = samples or 0