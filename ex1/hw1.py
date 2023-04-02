###### Your ID ######
# ID1: 204266191
# ID2: alian id
#####################

# imports 
import numpy as np
import pandas as pd

def preprocess(X,y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """
    ###########################################################################
    # TODO: Implement the normalization function.                             #
    ###########################################################################
    X = (X - np.mean(X, axis=0))/(np.max(X, axis=0) - np.min(X, axis=0))
    y = (y - np.mean(y, axis=0))/(np.max(y, axis=0) - np.min(y, axis=0))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return X, y

def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """
    ###########################################################################
    # TODO: Implement the bias trick by adding a column of ones to the data.                             #
    ###########################################################################
    X = np.hstack( # add columns to the left
                (np.ones((X.shape[0], 1)), X) # creat a columns of ones according to number of rows in X
                 )

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return X

def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """

    J = 0  # We use J for the cost.
    ###########################################################################
    # TODO: Implement the MSE cost function.                                  #
    ###########################################################################
    J = ((X @ theta - y) ** 2).mean(axis=0) / 2
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return J

#################### Auxilary function ####################
def _mse_partial_derivative_vectorized(theta, j, X, y):
    # This function takes vector of parameters (theta), index of parameter (j), train data (X)
    # and real value (y) and return the partial derivative of MSE by the j parameter of theta

    # number of instances
    m = X.shape[0]

    # Using the formula learned in calss for the partial derivative of the cost function
    return ((X @ theta - y) * X[:,j]).sum()/m
###########################################################

def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    ###########################################################################
    # TODO: Implement the gradient descent optimization algorithm.            #
    ###########################################################################
    # Initialize a temporary copy of theta
    current_theta = theta

    # Iterate "iterations" times
    for t in range(0, num_iters):

        # initialize a variable to keep the cost due to each theta
        theta_errors = np.zeros(theta.shape[0])

        # Apply the gradient descent algorithm for each parameter in theta separately
        for theta_i in enumerate(current_theta, start=0):

            # unpack the enumerate object to find the index of the current theta we derivate by
            j = theta_i[0]

            # compute the descent with an auxiliary function
            descent = _mse_partial_derivative_vectorized(theta, j, X, y)

            # keep the mse of theta j in order to compute J_history later
            theta_errors[j] = descent

            # update the value of the current theta by the descent we got modified by learning rate
            current_theta[j] = current_theta[j] - alpha * descent

        # update J_history with the current total descent
        J_history.append(theta_errors.sum())

        # update theta with the new value afte iteration "theta_i"
        theta = current_theta
        ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return theta, J_history

def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """
    
    pinv_theta = []
    ###########################################################################
    # TODO: Implement the pseudoinverse algorithm.                            #
    ###########################################################################
    pinv_theta = np.linalg.inv(np.transpose(X) @ X) @ np.transpose(X) @ y
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pinv_theta

def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    ###########################################################################
    # TODO: Implement the efficient gradient descent optimization algorithm.  #
    ###########################################################################
    current_theta = theta

    # Iterate "iterations" times
    for t in range(0, num_iters):

        # initialize a variable to keep the cost due to each theta
        theta_errors = np.zeros(theta.shape[0])

        # Apply the gradient descent algorithm for each parameter in theta separately
        for theta_i in enumerate(current_theta, start=0):

            # unpack the enumerate object to find the index of the current theta we derivate by
            j = theta_i[0]

            # compute the descent with an auxiliary function
            descent = _mse_partial_derivative_vectorized(theta, j, X, y)

            # in order to make it efficent we break from the loop if the error is smaller than 1e-8
            if descent < 1e-8:
                break

            # keep the mse of theta j in order to compute J_history later
            theta_errors[j] = descent

            # update the value of the current theta by the descent we got modified by learning rate
            current_theta[j] = current_theta[j] - alpha * descent

        # update J_history with the current total descent
        J_history.append(theta_errors.sum())

        # update theta with the new value afte iteration "theta_i"
        theta = current_theta

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return theta, J_history

def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """
    
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {} # {alpha_value: validation_loss}
    ###########################################################################
    # TODO: Implement the function and find the best alpha value.             #
    ###########################################################################

    # computing efficiently the theta vector's for each value of alpha on the train data
    thetas = [efficient_gradient_descent(X_train,
                               y_train,
                               np.transpose(np.ones(X_train.shape[1])), # picking arbitrary ones as initial values
                               alpha,
                               iterations)[0] for alpha in alphas]
    # computing the cost by the test data
    for theta, alpha in zip(thetas, alphas):
        alpha_dict[alpha] = compute_cost(X_val, y_val, theta)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return alpha_dict

def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []
    #####c######################################################################
    # TODO: Implement the function and find the best alpha value.             #
    ###########################################################################
    temp_costs = np.zeros(X_train[0])
    unselected_features = list(range(0, X_train[0]))
    current_thetas = {} # indices

    for i in range (1, 6):
        for j in unselected_features:
            if j not in selected_features:
                theta = efficient_gradient_descent(X_train[list(current_thetas.keys())],
                                                   y_train,
                                                   np.ones(i),
                                                   best_alpha,
                                                   iterations)[0]
                temp_costs[j] = compute_cost(X_val, y_val, current_thetas + [theta])
        current_thetas[-1] =
        selected_features.append(np.argmin(temp_costs))
        del unselected_features[np.argmin(temp_costs)]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return selected_features

def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()
    ###########################################################################
    # TODO: Implement the function to add polynomial features                 #
    ###########################################################################
    new_vars = {} # initialize an empty dictionary to store the combinations
    for i in df.columns:
        for j in df.columns:
            if (i, j) not in new_vars.keys() and (j, i) not in new_vars.keys(): #check if we allready have this variable
                if i == j:
                    new_vars[(i, j)] = j + "^2"
                else:
                    new_vars[(i, j)] = i + "*" + j

    for variables ,column_name in new_vars.items():
        df_poly[column_name] = df[variables[0]] * df[variables[1]]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return df_poly