import numpy as np

def sigmoid(x):
    """
    Compute the sigmoid function for a given input x.

    The sigmoid function, also known as the logistic function, is a non-linear
    activation function that maps any real-valued number to a value between 0 and 1.

    Parameters:
    x (numpy.ndarray or float): The input value or array.

    Returns:
    numpy.ndarray or float: The sigmoid value(s) corresponding to the input.
    """
    return 1 / (1 + np.exp(-x))

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

        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################

        self.theta = np.zeros((X.shape[1]) + 1)

        X = np.column_stack((np.ones(X.shape[0]), X))

        for i in range(self.n_iter):

            # update theta using gradient descent
            h = sigmoid(np.dot(X, self.theta))
            gradient = np.dot(X.T, (h - y)).reshape(-1,1)
            self.theta[:,np.newaxis] -= self.eta * gradient

            # store theta for each iteration
            self.thetas.append(self.theta.copy())

            # compute cost function and append to Js history
            J = (-1 / len(y)) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
            self.Js.append(J)

            # check convergence according to self.eps
            if len(self.Js) > 1 and abs(self.Js[-1] - self.Js[-2]) < self.eps:
                break

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

        preds = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################

        # Perform prediction for each instance in X using the learned model parameters
        # if the sigmoid output is greater than 0.5, assign the class label 1, else assign 0
        # store the predicted class labels in the 'preds' list

        # apply bias trick
        X = np.column_stack((np.ones(X.shape[0]), X))

        preds = [1 if sigmoid(np.dot(self.theta, x)) > 0.5 else 0 for x in X]

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

    cv_accuracy = None

    # set random seed
    np.random.seed(random_state)

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################

    all_indices = np.random.permutation(X.shape[0])
    shuffled_X = X[all_indices]
    shuffled_y = y[all_indices]

    folds_indices = np.array_split(all_indices, folds)

    accuracies = []

    for indices in folds_indices:

        test_indices = np.isin(np.arange(shuffled_X.shape[0]), indices)
        train_indices = ~ test_indices

        x_train = shuffled_X[train_indices]
        y_train = shuffled_y[train_indices]
        x_test = shuffled_X[test_indices]
        y_test = shuffled_y[test_indices]

        algo.fit(x_train,y_train)

        accuracies.append(np.mean(algo.predict(x_test) == y_test))

    cv_accuracy = np.mean(accuracies)

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
    p = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    p = (1 / (sigma * (2*np.pi) ** 0.5))*np.exp(-0.5*((data-mu)/sigma)**2)
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

        self.responsibilities = None
        self.weights = np.array([])
        self.mus = np.array([])
        self.sigmas = np.array([])
        self.costs = np.array([])

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.weights = np.ones(self.k) / self.k
        self.mus = np.random.rand(self.k, data.shape[1])
        self.sigmas = np.ones((self.k, data.shape[1]))
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def compute_cost(self, data, weights, mus, sigmas):
        m = data.shape[0]
        return np.sum([-np.log(np.sum(weights * norm_pdf(x, mus, sigmas))) for x in data])

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        m = data.shape[0]
        n = data.shape[1]
        self.responsibilities = np.zeros((m,n))

        denominator = np.sum(np.multiply(self.weights[i],norm_pdf(data, self.mus[i], self.sigmas[i])) for i in range(self.k))
        print("denominator: ", denominator)

        for i in range(self.k):
            numerator = self.weights * norm_pdf(data[i], self.mus, self.sigmas)
            denominator = np.sum(numerator)
            print(numerator.shape)
            print(denominator.shape)
            print(numerator)

            self.responsibilities[i] = numerator / denominator

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        N = data.shape[0]
        self.weights = 1
        self.mus = 1
        self.sigmas = 1

        self.weights = np.sum(self.responsibilities, axis=0) / N

        weighted_sum = np.dot(self.responsibilities.T, data)
        self.mus = weighted_sum / np.sum(self.responsibilities, axis=0, keepdims=True)

        squared_diff = (data - self.mus[:, np.newaxis]) ** 2
        self.sigmas = np.sqrt(
            np.sum(self.responsibilities[:, :, np.newaxis] * squared_diff, axis=0) / np.sum(self.responsibilities,
                                                                                            axis=0, keepdims=True))


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
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################

        self.init_params(data)
        cost_prev = 0

        for _ in range(self.n_iter):
            self.expectation(data)
            self.maximization(data)
            cost = self.compute_cost(data, self.weights, self.mus, self.sigmas)
            np.append(self.costs, cost)
            if abs(cost_prev - cost) < self.eps:
                break


        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas

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
    pdf = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pdf = np.sum([weight * norm_pdf(data, mu, sigma) for weight, mu, sigma in zip(weights, mus, sigmas)])
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
        self.random_state = random_state
        self.prior = None
        self.em = EM(k=k, random_state=random_state)

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
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################

        self.prior = np.zeros(2)
        self.em.fit(np.hstack(X, y))

        self.prior[0] = np.sum(y == 0) / len(y)
        self.prior[1] = np.sum(y == 1) / len(y)

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
        preds = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################

        preds = []
        weights, mus, sigmas = self.em.get_dist_params()

        for x in X:
            likelihoods = []
            for c in range(2):
                class_likelihood = 0.0
                for i in range(self.k):
                    class_likelihood += weights[i, c] * norm_pdf(x, mus[i, c], sigmas[i, c])
                likelihoods.append(class_likelihood)

            preds.append(np.argmax(likelihoods))
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

    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
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
    dataset_a_features = None
    dataset_a_labels = None
    dataset_b_features = None
    dataset_b_labels = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return{'dataset_a_features': dataset_a_features,
           'dataset_a_labels': dataset_a_labels,
           'dataset_b_features': dataset_b_features,
           'dataset_b_labels': dataset_b_labels
           }