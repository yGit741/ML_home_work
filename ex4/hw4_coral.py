import numpy as np

class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []

    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        # set random seed
        np.random.seed(self.random_state)

        X = np.array(X)
        y = np.array(y)
        self.J_hist = []
        X = np.column_stack((np.ones(len(X)), X))  # Stack ones column to X

        self.theta = np.random.random(X.shape[1])
        for i in range(self.n_iter):
            h = self.sigmoid(np.dot(X, self.theta))  # Use np.dot for matrix multiplication
            self.theta -= (self.eta / len(X)) * np.dot(X.T, (h - y))  # Use np.dot for matrix multiplication
            self.J_hist.append(self.compute_cost(X, y))
            if len(self.J_hist) > 1:
                if np.abs(self.J_hist[i] - self.J_hist[i - 1]) <= self.eps:
                    break
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def compute_cost(self,X,y):
      h = self.sigmoid(np.dot(X,self.theta))
      return (1 / len(X)) * ((np.dot((-y), np.log(h))) - (np.dot((1 - y), np.log(1-h))))
    
    def sigmoid(self,X):
      exponent = np.exp(X)
      return exponent / (1 + exponent)


    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        ones_column = np.ones((len(X), 1))
        x = np.hstack((ones_column, X))
        preds = np.round(self.sigmoid(np.dot(x,self.theta)))
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds

def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """

    # set random seed
    np.random.seed(random_state)

    # Shuffle the data
    indices = np.random.permutation(X.shape[0])
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    # Create fold indices
    fold_indices = np.array_split(indices, folds)
    fold_size = X.shape[0] // folds
    # Adjust the last fold size if necessary
    if len(fold_indices[-1]) < fold_size:
        fold_indices[-1] = np.concatenate(fold_indices[-1:], axis=None)

    accuracy_list = []
    for indices in fold_indices:
        # Split the data into training and validation sets
        X_train = X_shuffled[np.concatenate(fold_indices != indices)]
        y_train = y_shuffled[np.concatenate(fold_indices != indices)]
        X_val = X_shuffled[indices]
        y_val = y_shuffled[indices]

        # Train the model on the training set
        algo.fit(X_train, y_train)

        # Predict the labels for the validation set
        y_pred = algo.predict(X_val)

        # Calculate accuracy for the validation set
        accuracy = np.mean(y_pred == y_val)
        accuracy_list.append(accuracy)

    # Calculate cross-validation accuracy
    cv_accuracy = np.mean(accuracy_list)
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return cv_accuracy

def norm_pdf(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mu and sigma for the given x.    
    """
    p = (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((data - mu) / sigma) ** 2)
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return p

class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.responsibilities = {}
        self.weights = []
        self.mus = []
        self.sigmas = []
        self.costs = []

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """
        np.random.seed(self.random_state)
        self.weights = [1/self.k] * self.k
        max_mu = np.max(data)
        min_mu = np.min(data)
        self.mus = np.random.default_rng().uniform(min_mu,max_mu,self.k)
        self.sigmas = np.random.default_rng().uniform(1,2,self.k)
        
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        denominator = 0
        for i in range(self.k):
            denominator += np.multiply(self.weights[i],norm_pdf(data, self.mus[i], self.sigmas[i]))

        for i in range(self.k):
            nominator = np.multiply(self.weights[i],norm_pdf(data, self.mus[i], self.sigmas[i]))
            self.responsibilities[i] = nominator / denominator
            print(self.responsibilities[i])
        print(self.responsibilities)
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        for i in range(self.k):
          self.weights[i] = np.sum(self.responsibilities[i]) / len(data)
          self.mus[i] = np.sum(np.multiply(self.responsibilities[i],data)) / (np.multiply(self.weights[i],len(data)))
          self.sigmas[i] = np.sqrt(np.sum(np.multiply(self.responsibilities[i],np.square(data - self.mus[i]))) / (np.multiply(self.weights[i],len(data))))

        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        self.init_params(data)
        self.costs = []

        for iteration in range(self.n_iter):
            self.expectation(data)
            self.maximization(data)
            self.costs.append(self.cost_function(data))

            if len(self.costs) > 1:
              if np.abs(self.costs[-1] - self.costs[-2]) < self.eps:
                break

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas

    def cost_function(self, data):
        cost = 0
        for i in range(self.k):
            cost += np.sum(-np.log(np.multiply(self.weights[i],norm_pdf(data, self.mus[i], self.sigmas[i]))))
        return cost

def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.
 
    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.    
    """
    pdf = [] 
    for i in range(len(mus)):
        pdf.append(np.multiply(weights[i],norm_pdf(data,mus[i], sigmas[i])))
    
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pdf

class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.params = {}
        self.priors = []
        self.em = EM(self.k)
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        classes = np.unique(y)
        dataset = np.hstack((X, y.reshape(-1, 1)))
        for val in classes:
            self.params[val] = {}
            cls = dataset[dataset[:, -1] == val]
            prior = len(cls) / len(dataset)
            self.params[val]['prior'] = prior

            self.em.fit(cls[:, :-1])
            self.params[val]['means'], self.params[val]['stds'], self.params[val]['weights'] = self.em.get_dist_params()
    
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        classes = list(self.params.keys())

        post = np.ones((len(X), len(classes)))
        for i, val in enumerate(classes):
            means = self.params[val]['means']
            stds = self.params[val]['stds']
            weights = self.params[val]['weights']

            likelihood = np.prod(weights * norm_pdf(X, means, stds), axis=1)
            post[:, i] = likelihood * self.params[val]['prior']

        preds = np.argmax(post, axis=1)

        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds

def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    ''' 
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    ''' 

    lor_model = LogisticRegressionGD(eta=best_eta, eps=best_eps)
    lor_model.fit(x_train, y_train)

    # Predict using Logistic Regression model
    lor_predict = lor_model.predict(x_test)

    # Calculate accuracy for Logistic Regression model on train and test sets
    lor_train_acc = np.mean(lor_model.predict(x_train) == y_train)
    lor_test_acc = np.mean(lor_predict == y_test)

    naive_model = NaiveBayesGaussian(k=k)
    naive_model.fit(x_train, y_train)

    # Predict using Naive Bayes model
    naive_predict = naive_model.predict(x_test)

    # Calculate accuracy for Naive Bayes model on train and test sets
    bayes_train_acc = np.mean(naive_model.predict(x_train) == y_train)
    bayes_test_acc = np.mean(naive_predict == y_test)

    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}

def generate_datasets():
    from scipy.stats import multivariate_normal
    '''
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    '''
    # Define parameters for class 1
    mean_c1 = np.array([1, 1, 1])
    cov_c1 = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])

    # Define parameters for class 2
    mean_c2 = np.array([-1, -1, -1])
    cov_c2 = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])

    # Generate samples for class 1
    dataset_a = multivariate_normal.rvs(mean=mean_c1, cov=cov_c1, size=1000)

    # Generate samples for class 2
    dataset_b = multivariate_normal.rvs(mean=mean_c2, cov=cov_c2, size=1000)

    dataset_a_features = np.concatenate([dataset_a,dataset_b],axis=0)
    dataset_a_labels = np.array([0] * 1000 + [1] * 1000)

    # Define parameters for class 1
    mean_c1 = np.array([0, 0, 0])
    cov_c1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # Define parameters for class 2
    mean_c2 = np.array([2, 2, 2])
    cov_c2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # Generate samples for class 1
    dataset_a = multivariate_normal.rvs(mean=mean_c1, cov=cov_c1, size=1000)

    # Generate samples for class 2
    dataset_b = multivariate_normal.rvs(mean=mean_c2, cov=cov_c2, size=1000)

    dataset_b_features = np.concatenate([dataset_a,dataset_b],axis=0)
    dataset_b_labels = np.array([0] * 1000 + [1] * 1000)

    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return{'dataset_a_features': dataset_a_features,
           'dataset_a_labels': dataset_a_labels,
           'dataset_b_features': dataset_b_features,
           'dataset_b_labels': dataset_b_labels}
