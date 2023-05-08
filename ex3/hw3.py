import numpy as np

class conditional_independence():

    def __init__(self):

        # You need to fill the None value with *valid* probabilities
        self.X = {0: 0.3, 1: 0.7}  # P(X=x)
        self.Y = {0: 0.3, 1: 0.7}  # P(Y=y)
        self.C = {0: 0.5, 1: 0.5}  # P(C=c)

        self.X_Y = {
            (0, 0): 0.1,
            (0, 1): 0.2,
            (1, 0): 0.2,
            (1, 1): 0.5
        }  # P(X=x, Y=y)

        self.X_C = {
            (0, 0):  0.15,
            (0, 1): 0.15,
            (1, 0): 0.35,
            (1, 1):  0.35
        }  # P(X=x, C=y)

        self.Y_C = {
            (0, 0):  0.15,
            (0, 1): 0.15,
            (1, 0): 0.35,
            (1, 1):  0.35
        }  # P(Y=y, C=c)

        self.X_Y_C = {
            (0, 0, 0): 0.09,
            (0, 0, 1): 0.09,
            (0, 1, 0): 0.21,
            (0, 1, 1): 0.21,
            (1, 0, 0): 0.21,
            (1, 0, 1): 0.21,
            (1, 1, 0): 0.49,
            (1, 1, 1): 0.49,
        }  # P(X=x, Y=y, C=c)

    def is_X_Y_dependent(self):
        """
        return True iff X and Y are depndendent
        """
        X = self.X
        Y = self.Y
        X_Y = self.X_Y
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################

        # initialize a boolean that naively assume independency
        dependent = True

        # iterate over X support
        for x in X.keys():

            # iterate over Y support
            for y in Y.keys():

                # if there exist x,y such that P(X=x,Y=y) =! P(X=x)*P(Y=y) there is
                # dependency and return is_dependent
                if not np.isclose(self.X[x] * self.Y[y], X_Y[(x,y)]):
                    return dependent

        # all probabilities of P(X=x, Y=y) equal to the multiplications, therefore
        # return the original boolean not dependent
        return not dependent
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def is_X_Y_given_C_independent(self):
        """
        return True iff X_given_C and Y_given_C are indepndendent
        """
        X = self.X
        Y = self.Y
        C = self.C
        X_C = self.X_C
        Y_C = self.Y_C
        X_Y_C = self.X_Y_C
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        X_Y = self.X_Y

        # initialize a boolean that naively assume independency
        cond_independent = True

        # iterate over X support
        for x in X.keys():
            # print("X = ", str(x))

            # iterate over Y support
            for y in Y.keys():

                # print("\t Y = ", str(y))

                # iterate over C support
                for c in C.keys():
                    # print("\t\t C = ", str(c))
                    # if there exist x,y,c such that P(X=x,Y=y|C=c) =! P(X=x|C=c)*P(Y=y|C=c)
                    # there is conditional dependency and return False
                    #
                    # print(f"\t\t P(X = {x}) = ", X[x])
                    # print(f"\t\t P(Y = {y}) = ", Y[y])
                    # print(f"\t\t P(C = {c}) = ", C[c])
                    # print(f"\t\t P(X = {x}, Y = {y}) = ",X_Y[(x, y)])
                    # print(f"\t\t P(X = {x}, C = {c}) = ",X_C[(x, c)])
                    # print(f"\t\t P(Y = {y}, C = {c}) = ",Y_C[(y, c)])
                    # print(f"\t\t P(X = {x} | C = {c}) = ",X_C[(x, c)] / C[c])
                    # print(f"\t\t P(Y = {y} | C = {c}) = ",Y_C[(y, c)] / C[c])
                    # print(f"\t\t P(X = {x}, Y = {y}, C = {c}) = ", X_Y_C[(x, y, c)])
                    #
                    # print("****************")
                    # print(X_Y_C[x,y,c], " ", (X_C[x,c]/C[c])*(Y_C[y,c]/C[c]))
                    # print("****************")

                    if not np.isclose(X_Y_C[x,y,c],(X_C[x,c]/C[c])*(Y_C[y,c]/C[c])):
                        return not cond_independent


        # all probabilities of P(X=x, Y=y) equal to the multiplications, therefore
        # return the original is_independent as True
        return cond_independent

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

def poisson_log_pmf(k, rate):
    """
    k: A discrete instance
    rate: poisson rate parameter (lambda)

    return the log pmf value for instance k given the rate
    """
    log_p = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################

    log_p = np.log((rate ** k) * np.e ** (-rate) / np.math.factorial(k))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return log_p

def get_poisson_log_likelihoods(samples, rates):
    """
    samples: set of univariate discrete observations
    rates: an iterable of rates to calculate log-likelihood by.

    return: 1d numpy array, where each value represent that log-likelihood value of rates[i]
    """
    likelihoods = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################

    likelihoods = [np.sum([poisson_log_pmf(k, rate) for k in samples]) for rate in rates]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return likelihoods

def possion_iterative_mle(samples, rates):
    """
    samples: set of univariate discrete observations
    rate: a rate to calculate log-likelihood by.

    return: the rate that maximizes the likelihood 
    """
    rate = 0.0
    likelihoods = get_poisson_log_likelihoods(samples, rates) # might help
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################

    rate = rates[np.argmax(likelihoods)]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return rate

def possion_analytic_mle(samples):
    """
    samples: set of univariate discrete observations

    return: the rate that maximizes the likelihood
    """
    mean = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    mean = np.mean(samples)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return mean

def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and std for the given x.    
    """
    p = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    assert std > 0, "Invalid input in normal_pdf - standard deviation must be positive"
    assert x is not None and mean is not None and std is not None, "Error: All input arguments must be provided"

    coefficient = 1 / (np.sqrt(2 * np.pi * std **2))
    exponent = np.e**(-0.5*((x - mean)/std)**2)
    p = coefficient * exponent

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return p

class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulates the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.
        
        Input
        - dataset: The dataset as a 2d numpy array, assuming the class label is the last column
        - class_value : The class to calculate the parameters for.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################

        self.full_dataset = dataset
        self.data = dataset[dataset[:,-1] == class_value]
        self.mu = np.mean(self.data[:,:-1], axis=0)
        self.sigma = np.std(self.data[:,:-1], axis=0)

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def get_prior(self):
        """
        Returns the prior probability of the class according to the dataset distribution.
        """
        prior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################

        prior = self.data.shape[0] / self.full_dataset.shape[0]

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood probability of the instance under the class according to the dataset distribution.
        """
        likelihood = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # print(x[0])

        likelihood = np.prod([normal_pdf(x[j], self.mu[j], self.sigma[j]) for j in range(self.data.shape[1] - 1)])

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        posterior = self.get_instance_likelihood(x) * self.get_prior()

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return posterior

class MAPClassifier():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions. 
        One for class 0 and one for class 1, and will predict an instance
        using the class that outputs the highest posterior probability 
        for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods 
                     for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods 
                     for the distribution of class 1.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################

        self.ccd0 = ccd0

        self.ccd1 = ccd1


        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        if self.ccd0.get_instance_posterior(x) > self.ccd1.get_instance_posterior(x):
            pred = 0
        else:
            pred = 1

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred

def compute_accuracy(test_set, map_classifier):
    """
    Compute the accuracy of a given a test_set using a MAP classifier object.
    
    Input
        - test_set: The test_set for which to compute the accuracy (Numpy array). where the class label is the last column
        - map_classifier : A MAPClassifier object capable of predicting the class for each instance in the test set.
        
    Ouput
        - Accuracy = #Correctly Classified / test_set size
    """
    acc = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    acc = np.mean([map_classifier.predict(test_set[i]) for i in range(test_set.shape[0])] == test_set[:,-1])

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return acc

def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variable normal desnity function for a given x, mean and covarince matrix.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean vector of the distribution.
    - cov:  The covariance matrix of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    pdf = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################

    # verify positive semidefinite matrix
    assert np.all(np.linalg.eigvals(cov) >= 0),  "Invalid covariance matrix - not positive semidefinite matrix. "

    # the dimension as integer
    d = cov.shape[0]

    # the inverse of the covaraince matrix
    inv_cov = np.linalg.inv(cov)

    # the subtraction of x from the mean as vector
    # print(mean)
    # print(x)
    x_mean = x - mean

    # print(x_mean.T)
    # print(inv_cov)
    # print(x_mean)
    # compute the pdf of the multivariate normal
    pdf = ((2*np.pi)**(-d/2))*(np.linalg.det(cov)**(-0.5))*np.exp(-0.5 * x_mean.T @ inv_cov @ x_mean)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pdf

class MultiNormalClassDistribution():

    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.
        
        Input
        - dataset: The dataset as a numpy array
        - class_value : The class to calculate the parameters for.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.full_dataset = dataset
        self.data = dataset[dataset[:,-1] == class_value]
        self.mu = np.mean(self.data[:,:-1], axis=0)
        self.sigma = np.cov(self.data[:,0], self.data[:,1])


        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################

        prior = self.data.shape[0] / self.full_dataset.shape[0]

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under the class according to the dataset distribution.
        """
        likelihood = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # print('start')
        # print(x[0], self.mu[0], self.sigma)
        # print(np.prod([multi_normal_pdf(x[j], self.mu[j], self.sigma) for j in range(self.data.shape[1] - 1)]))
        likelihood = multi_normal_pdf(x[:-1], self.mu, self.sigma)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################

        posterior = self.get_instance_likelihood(x) * self.get_prior()


        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return posterior

class MaxPrior():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum prior classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest prior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################

        self.ccd0 = ccd0
        self.ccd1 = ccd1

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        if self.ccd0.get_prior() > self.ccd1.get_prior():
            pred = 0
        else:
            pred = 1

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred

class MaxLikelihood():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum Likelihood classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest likelihood probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.ccd0 = ccd0
        self.ccd1 = ccd1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################

        if self.ccd0.get_instance_likelihood(x) > self.ccd1.get_instance_likelihood(x):
            pred = 0
        else:
            pred = 1

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred

EPSILLON = 1e-6 # if a certain value only occurs in the test set, the probability for that value will be EPSILLON.

class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes 
        distribution for a specific class. The probabilites are computed with laplace smoothing.
        
        Input
        - dataset: The dataset as a numpy array.
        - class_value: Compute the relevant parameters only for instances from the given class.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def get_prior(self):
        """
        Returns the prior porbability of the class 
        according to the dataset distribution.
        """
        prior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under 
        the class according to the dataset distribution.
        """
        likelihood = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return likelihood
        
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance 
        under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return posterior

class MAPClassifier_DNB():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predict an instance
        by the class that outputs the highest posterior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred

    def compute_accuracy(self, test_set):
        """
        Compute the accuracy of a given a testset using a MAP classifier object.

        Input
            - test_set: The test_set for which to compute the accuracy (Numpy array).
        Ouput
            - Accuracy = #Correctly Classified / #test_set size
        """
        acc = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return acc


