import numpy as np


def get_random_centroids(X, k):
    '''
    Each centroid is a point in RGB space (color) in the image.
    This function should uniformly pick `k` centroids from the dataset.
    Input: a single image of shape `(num_pixels, 3)` and `k`, the number of centroids.
    Notice we are flattening the image to a two dimentional array.
    Output: Randomly chosen centroids of shape `(k,3)` as a numpy array.
    '''
    centroids = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################

    random_indices = np.random.choice(X.shape[0], size=k, replace=False)

    centroids = X[random_indices]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # make sure you return a numpy array
    return np.asarray(centroids).astype(np.float64)


def lp_distance(X, centroids, p=2):
    '''
    Inputs: 
    A single image of shape (num_pixels, 3)
    The centroids (k, 3)
    The distance parameter p

    output: numpy array of shape `(k, num_pixels)` thats holds the distances of 
    all points in RGB space from all centroids
    '''
    distances = []
    k = len(centroids)
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################

    distances = np.sum(np.abs(X[:, np.newaxis] - centroids) ** p, axis=2)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return distances


def kmeans(X, k, p ,max_iter=100):
    """
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = []
    centroids = get_random_centroids(X, k)
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################

    # initialize empty numpy array to store the classes of the points
    classes = np.zeros(X.shape[0], dtype=int)

    for _ in range(max_iter):

        # store thr previous centroids
        old_centroids = centroids.copy()

        # calculate the distance between the points to centroid
        distances = lp_distance(X, centroids, p)

        # assign each point to its closet centroid
        classes = np.argmin(distances, axis=1)

        for i in range(k):

            # select all points assigned to centroid i
            cluster_points = X[classes == i]

            # verify that there is any points at this centroid
            if len(cluster_points) > 0:

                # update the centroid position
                centroids[i] = np.mean(cluster_points, axis=0)

        # check convergence of the centroids
        if np.all(old_centroids == centroids):
            break

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return centroids, classes


def kmeans_pp(X, k, p ,max_iter=100):
    """
    Your implenentation of the kmeans++ algorithm.
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = None
    centroids = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################

    centroids = get_distributed_centroids(X, k, p)
    classes = np.zeros(X.shape[0], dtype=int)

    for _ in range(max_iter):

        # store thr previous centroids
        old_centroids = centroids.copy()

        # calculate the distance between the points to centroid
        distances = lp_distance(X, centroids, p)

        # assign each point to its closet centroid
        classes = np.argmin(distances, axis=1)

        for i in range(k):

            # select all points assigned to centroid i
            cluster_points = X[classes == i]

            # verify that there is any points at this centroid
            if len(cluster_points) > 0:
                # update the centroid position
                centroids[i] = np.mean(cluster_points, axis=0)

        # check convergence of the centroids
        if np.all(old_centroids == centroids):
            break

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return centroids, classes


def get_distributed_centroids(X, k, p=2):

    # draw randomly a centroid among all data points
    centroid_0 = X[np.random.randint(0, X.shape[0])]

    # create a list to store al centroids (need to be compatible with lp_distance)
    centroids = np.zeros((k,3))
    centroids[0] = np.array([centroid_0])

    # initialize an empty
    distances = np.zeros((X.shape[0], k))
    current_centroid = centroid_0

    # choose all the other centroids based on the provided algorithm
    for i in range(1, k):

        # calculate the distances between the current centroid and all data points
        cur_distances = lp_distance(X, current_centroid,p=p)

        # compute the probabilities for selecting the next centroid based on distances
        prob = cur_distances / np.sum(cur_distances)

        # select the next centroid randomly based on the probabilities
        current_centroid = X[np.random.choice(X.shape[0], p=prob.flatten())]

        # store the selected centroid in the centroids array
        centroids[i] = current_centroid

    return centroids

def inertia(X, centroids, p):
    square_d = (lp_distance(X, centroids, p))**2
    print(square_d)
    minimal_d = np.min(square_d, axis=0)
    print(minimal_d)
    return np.sum(minimal_d)