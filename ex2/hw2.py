import numpy as np
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################

    _, counts = np.unique(data[:, -1], return_counts=True)

    gini = 1 - np.sum((counts / data.shape[0])**2)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################

    _, counts = np.unique(data[:, -1], return_counts=True)

    p_i = counts / data.shape[0]

    entropy = - (p_i * np.log2(p_i)).sum()

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return entropy

def goodness_of_split(data, feature, impurity_func, gain_ratio=False):
    """
    Calculate the goodness of split of a dataset given a feature and impurity function.
    Note: Python support passing a function as arguments to another function
    Input:
    - data: any dataset where the last column holds the labels.
    - feature: the feature index the split is being evaluated according to.
    - impurity_func: a function that calculates the impurity.
    - gain_ratio: goodness of split or gain ratio flag.

    Returns:
    - goodness: the goodness of split value
    - groups: a dictionary holding the data after splitting 
              according to the feature values.
    """
    goodness = 0
    groups = {} # groups[feature_value] = data_subset
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################

    # extract the unique values of the desired feature to split into groups (assume np.array to support np.unique)
    feature_values = np.unique(data[:, feature])

    # splitting the data into smaller data sets based on the feature with dictionary comprehension
    groups = {feature_char: data[data[:, feature] == feature_char] for feature_char in feature_values}

    # compute the impurity of the data according to impurity_func
    impurity = impurity_func(data)

    # calculate the goodness of split
    goodness = impurity - np.sum([(group.shape[0] / data.shape[0]) * impurity_func(group) for group in groups.values()])

    # compute the gain ratio if gain_ratio flag is True
    if gain_ratio:

        # calculate the split information
        split_info = calc_entropy(np.column_stack([data[:,feature]]))

        # assign o to goodness if split information is 0 to avoid dividing by 0
        if split_info ==0:
            goodness = 0
        else:
            goodness = goodness / split_info

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return goodness, groups

class DecisionNode:

    def __init__(self, data, feature=-1,depth=0, chi=1, max_depth=1000, gain_ratio=False):
        
        self.data = data # the relevant data for the node
        self.feature = feature # column index of criteria being tested
        self.pred = self.calc_node_pred() # the prediction of the node
        self.depth = depth # the current depth of the node
        self.children = [] # array that holds this nodes children
        self.children_values = []
        self.terminal = False # determines if the node is a leaf
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio 

        # get the number of features in the dataset
        self.num_features = self.data.shape[1] - 1
        self.full_data = self.data

    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################

        # checks that there are any rows in the data
        if self.data.shape[0] > 0:

            # return the labels and their counts
            unique_labels, counts = np.unique( self.data[:,-1], return_counts=True)

            # return the label with the highest count
            pred = unique_labels[np.argmax(counts)]

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred
        
    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        self.children.append(node)
        self.children_values.append(val)
     
    def split(self, impurity_func):

        """
        Splits the current node according to the impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to chi and max_depth.

        Input:
        - The impurity function that should be used as the splitting criteria

        This function has no return value
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # Check if current depth is greater than max_depth
        if self.depth >= self.max_depth:
            self.prune()
            return

        # calculate the goodness of split and groups and unpack them into separate variables
        gains, groups = zip(
                            *[goodness_of_split(self.data, feature, impurity_func, gain_ratio=self.gain_ratio) for
                              feature in range(self.num_features)]
                            )

        # check if the best goodness of split is zero, and if so initiate a leaf
        if max(gains) == 0:
            self.prune()
            return

        # if all(gain == 0 for gain in gains):
        #     self.prune()
        #     return


        # get the best feature according to the calculated gains
        self.feature = np.argmax(gains)

        # returns all the unique values for the split feature and convert to list
        unique_values_split_feature = np.unique(self.full_data[:, self.feature])

        # split the data based on the unique values of the split feature with dictionary comprehension and assign it
        # into groups
        groups = {val: self.data[self.data[:, self.feature] == val] for val in unique_values_split_feature}

        if len(unique_values_split_feature) == 1:
            assert False, "only one child!!!"


        for value in unique_values_split_feature:

            # create a new node with the corresponding groups
            current_node = DecisionNode(groups[value],
                                      feature=-1,
                                      depth=self.depth + 1,
                                      chi=self.chi,
                                      max_depth=self.max_depth,
                                      gain_ratio=self.gain_ratio)

            # add the children to the children list
            self.add_child(current_node, value)

        # check if the chi stat is less than the threshold and if so, set the node as a terminal node and return None
        # without adding children_list to the node.
        if 0 < self.chi < 1:

            # get the degree of freedom for the chi square test
            degree_of_freedom = len(self.children_values)

            # calculate the threshold for chi square test according to the global variable chi_table
            threshold = chi_table[degree_of_freedom][self.chi]

            # calculate the chi stat for the current node
            chi_stat = self.calc_chi_stat(groups)

            # check if chi stat is less than the threshold and if so, set the node as a terminal node and return None
            if chi_stat < threshold:
                self.prune()
                return

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def prune(self):
        self.children = []
        self.children_values = []
        self.terminal = True

    def perfectly_classified(self):

        """
        Checks if the data in the node is perfectly classified, which mean that all the target values are equal.

        Input:
        - self: the current node.

        Output:
        - True if the data in the node is perfectly classified, False otherwise.
        """
        target = self.data[:, -1]

        return np.all(target[0] == target)

    def calc_chi_stat(self, group):
        PY_0 = self.data[self.data[:, -1] == 'e'].shape[0] / self.data.shape[0]
        PY_1 = self.data[self.data[:, -1] == 'p'].shape[0] / self.data.shape[0]
        chi_stat = 0
        for f, group in group.items():
            pf = group[group[:, -1] == 'e'].shape[0]
            nf = group[group[:, -1] == 'p'].shape[0]
            E0 = PY_0 * (pf + nf)
            E1 = PY_1 * (pf + nf)
            chi_stat += ((pf - E0) ** 2) / E0 + ((nf - E1) ** 2) / E1
        return chi_stat

    def calc_depth(self):
        if not self.children:
            return self.depth
        return max([child.calc_depth() for child in self.children])


def build_tree(data, impurity, gain_ratio=False, chi=1, max_depth=1000):
    """
    Build a tree using the given impurity measure and training dataset. 
    You are required to fully grow the tree until all leaves are pure unless
    you are using pruning

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.
    - gain_ratio: goodness of split or gain ratio flag

    Output: the root node of the tree.
    """
    root = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################

    # construct a root node with the given data
    root = DecisionNode(data, feature=-1, depth=0, chi=chi, max_depth=max_depth, gain_ratio=gain_ratio)

    # enqueue the root
    queue = [root]

    while len(queue) > 0:

        # pop the last  node in the queue and assign it into node variable
        node = queue.pop()

        # split the node by its best feature according to the chosen impurity measure
        node.split(impurity)

        # add the children of the current node to the queue
        for child in node.children:
            queue.append(child)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return root

def predict(root, instance):
    """
    Predict a given instance using the decision tree
 
    Input:
    - root: the root of the decision tree.
    - instance: an row vector from the dataset. Note that the last element 
                of this vector is the label of the instance.
 
    Output: the prediction of the instance.
    """
    pred = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################

    # initialize variable that point to the current node with the root argument
    current_node = root

    # loop down the tree until we get to a leaf
    while not current_node.terminal:

        # value of the feature of current_node at the given instance
        instance_value_at_current_feature = instance[current_node.feature]

        if instance_value_at_current_feature in current_node.children_values:
            # find the index of the children that his value is instance_value_at_current_feature
            index_of_relevant_children = current_node.children_values.index(instance_value_at_current_feature)

            # assign the next node down the tree from the children attribute by index_of_relevant_children
            current_node = current_node.children[index_of_relevant_children]

        else:
            pred = current_node.calc_node_pred()
            return pred

    # return the predicted value of the tree to the instance
    pred = current_node.calc_node_pred()

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pred

def calc_accuracy(node, dataset):
    """
    Predict a given dataset using the decision tree and calculate the accuracy
 
    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated
 
    Output: the accuracy of the decision tree on the given dataset (%).
    """
    accuracy = 0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################

    # extract the values of the target feature that we evaluate accuracy by
    test_val = dataset[:,-1]

    # extract the number of instances of the dataset
    instances = dataset.shape[0]

    # use the node and the predict function to predict value for all instances in the dataset
    # predicted_vals = [predict(node, instance) for instance in dataset]

    predicted_vals = []
    for instance in dataset:
        predicted_vals.append(predict(node, instance))

    # calculate the acuuracy of the model in the node
    accuracy = np.sum(predicted_vals == test_val) / instances

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return accuracy

def depth_pruning(X_train, X_test):
    """
    Calculate the training and testing accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels
 
    Output: the training and testing accuracies per max depth
    """
    training = []
    testing  = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:

        current_tree = build_tree(X_train, calc_gini, max_depth=max_depth)
        training.append(calc_accuracy(current_tree, X_train))
        testing.append(calc_accuracy(current_tree, X_test))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return training, testing


def chi_pruning(X_train, X_test):

    """
    Calculate the training and testing accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_testing_acc: the testing accuracy per chi value
    - depths: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_testing_acc  = []
    depth = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################

    for p_val in [1, 0.5, 0.25, 0.1, 0.05, 0.0001]:

        current_tree = build_tree(X_train, calc_gini, chi=p_val)
        chi_training_acc.append(calc_accuracy(current_tree, X_train))
        chi_testing_acc.append(calc_accuracy(current_tree, X_test))
        depth.append(current_tree.calc_depth())

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return chi_training_acc, chi_testing_acc, depth


def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of nodes in the tree.
    """
    n_nodes = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################

    n_nodes = count_nodes_recursively(node)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return n_nodes


def count_nodes_recursively(node):
    """
    Count the number of node in a given tree using recursion. This auxiliary function is required due to the constraint
    that n_nodes is assigned to be None type which interrupt the natural implementation with recurssion.

    Input:
    - node: a node in the decision tree.

    Output: the number of nodes in the tree.
    """

    # initialize 1 in n_nodes for the current node
    n_nodes = 1

    # check if  node have any children
    if node.children:

        # iterate over all children of the node, call recursively count_nodes_recursively and add the output to n_nodes
        for child in node.children:
            n_nodes += count_nodes_recursively(child)

    # return n_nodes
    return n_nodes

