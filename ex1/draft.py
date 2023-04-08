def c1_forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
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
    for _ in range(0,5):
        current_costs = []
        for i in range(X_train.shape[1]):
            current_features = np.concatenate([selected_features, [i]]).astype(int)
            current_data = X_train[:,current_features]
            current_theta = efficient_gradient_descent(current_data,
                                                     y_train,
                                                     np.ones((current_data.shape[1], 1)),
                                                     best_alpha,
                                                     iterations)[0]
            cost = compute_cost(current_data,y_val.reshape(-1, 1),current_theta)
            current_costs.append(cost)
        # Find the index of the minimum cost among the current feature set
        best_feature = np.argmin(current_costs)
        # Add the selected feature to the list of selected features
        selected_features.append(best_feature)
        # Remove the selected feature from the set of candidate features
        remaining_features = np.setdiff1d(np.arange(X_train.shape[1]), selected_features)
        # If there are no remaining features, stop iterating
        if len(remaining_features) == 0:
            break
        # Otherwise, update X_train to include the selected features
        X_train = X_train[:, np.concatenate([selected_features, remaining_features])]
    return selected_features

def c2_forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
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
    X_train = apply_bias_trick(X_train)
    X_val = apply_bias_trick(X_val)

    selected_features = []

    for _ in range(5):
        current_costs = []
        current_features = []
        for i in range(X_train.shape[1]):
            if i not in selected_features:
                current_features.append(i)
                current_data = X_train[:, current_features]
                print("theta ", np.ones((current_data.shape[1], 1)).shape)
                current_theta = efficient_gradient_descent(current_data,
                                                           y_train,
                                                           np.ones((current_data.shape[1], 1)),
                                                           best_alpha,
                                                           iterations)[0]
                cost = compute_cost(current_data, y_val, current_theta)
                current_costs.append(cost)
                current_features.pop()

        best_feature = np.argmin(current_costs)
        selected_features.append(best_feature)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return selected_features




############################ delete before submission##########################

# df = pd.read_csv('data.csv')
# feature_names = df.drop(columns=['price', 'id', 'date']).columns.values
# X = df.drop(columns=['price', 'id', 'date']).values
# y = df['price'].values
#
# # preprocessing
# X, y = preprocess(X, y)
#
# # training and validation split
# np.random.seed(42)
# indices = np.random.permutation(X.shape[0])
# idx_train, idx_val = indices[:int(0.8*X.shape[0])], indices[int(0.8*X.shape[0]):]
# X_train, X_val = X[idx_train,:], X[idx_val,:]
# y_train, y_val = y[idx_train], y[idx_val]

############################ delete before submission##########################

# ffs = forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, 100)
# for feature in ffs:
#     print(feature_names[feature])
