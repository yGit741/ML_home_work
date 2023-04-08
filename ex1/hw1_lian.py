#!/usr/bin/env python
# coding: utf-8

# # Exercise 1: Linear Regression
# 
# ## Do not start the exercise until you fully understand the submission guidelines.
# 
# 
# * The homework assignments are executed automatically. 
# * Failure to comply with the following instructions will result in a significant penalty. 
# * Appeals regarding your failure to read these instructions will be denied. 
# * Kindly reminder: the homework assignments contribute 50% of the final grade.
# 
# 
# ## Read the following instructions carefully:
# 
# 1. This Jupyter notebook contains all the step-by-step instructions needed for this exercise.
# 1. Write **efficient**, **vectorized** code whenever possible. Some calculations in this exercise may take several minutes when implemented efficiently, and might take much longer otherwise. Unnecessary loops will result in point deductions.
# 1. You are responsible for the correctness of your code and should add as many tests as you see fit to this jupyter notebook. Tests will not be graded nor checked.
# 1. Complete the required functions in `hw1.py` script only. This exercise is graded automatically, and only the `hw1.py` script is tested.
# 1. You are allowed to use functions and methods from the [Python Standard Library](https://docs.python.org/3/library/), numpy and pandas only. **Do not import anything else.**
# 1. Your code must run without errors. Use at least `numpy` 1.15.4. Any code that cannot run will not be graded.
# 1. Write your own code. Cheating will not be tolerated.
# 1. Submission includes a zip file that contains the hw1.py script as well as this notebook, with your ID as the file name. For example, `hw1_123456789_987654321.zip` if you submitted in pairs and `hw1_123456789.zip` if you submitted the exercise alone.
# Please use only a **zip** file in your submission.
# 
# ---
# ---
# 
# ## Please sign that you have read and understood the instructions: 
# 
# ### *** YOUR ID HERE ***
# 
# ---
# ---

# In[1]:


import numpy as np # used for scientific computing
import pandas as pd # used for data analysis and manipulation
import matplotlib.pyplot as plt # used for visualization and plotting
np.random.seed(42) 

# make the notebook automatically reload external python modules
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ## Part 1: Data Preprocessing (10 Points)
# 
# For the following exercise, we will use a dataset containing housing prices in King County, USA. The dataset contains 5,000 observations with 18 features and a single target value - the house price. 
# 
# First, we will read and explore the data using pandas and the `.read_csv` method. Pandas is an open source library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language.

# In[2]:


# Read comma separated data
df = pd.read_csv('data.csv')
# df stands for dataframe, which is the default format for datasets in pandas


# ### Data Exploration
# A good practice in any data-oriented project is to first try and understand the data. Fortunately, pandas is built for that purpose. Start by looking at the top of the dataset using the `df.head()` command. This will be the first indication that you read your data properly, and that the headers are correct. Next, you can use `df.describe()` to show statistics on the data and check for trends and irregularities.

# In[3]:


df.head(5)


# In[4]:


df.describe()


# We will start with one variable linear regression by extracting the target column and the `sqft_living` variable from the dataset. We use pandas and select both columns as separate variables and transform them into a numpy array.

# In[5]:


X = df['sqft_living'].values
y = df['price'].values


# In[34]:


X


# In[35]:


y


# ## Preprocessing
# 
# As the number of features grows, calculating gradients gets computationally expensive. We can speed this up by normalizing the input data to ensure all values are within the same range. This is especially important for datasets with high standard deviations or differences in the ranges of the attributes. Use [mean normalization](https://en.wikipedia.org/wiki/Feature_scaling) for the fearures (`X`) and the true labels (`y`). 
# 
# Your implementation should not contain loops.
# 
# ---
# Open `hw1.py` and complete the function `preprocess`. (5 points)

# In[7]:


from hw1 import preprocess

X, y = preprocess(X, y)
X.shape, y.shape


# In[8]:


X


# In[9]:


y


# We will split the data into two datasets: 
# 1. The training dataset will contain 80% of the data and will always be used for model training.
# 2. The validation dataset will contain the remaining 20% of the data and will be used for model evaluation. For example, we will pick the best alpha and the best features using the validation dataset, while still training the model using the training dataset.

# In[10]:


# training and validation split
np.random.seed(42)
indices = np.random.permutation(X.shape[0])
idx_train, idx_val = indices[:int(0.8*X.shape[0])], indices[int(0.8*X.shape[0]):]
X_train, X_val = X[idx_train], X[idx_val]
y_train, y_val = y[idx_train], y[idx_val]


# ## Data Visualization
# Another useful tool is data visualization. Since this problem has only two parameters, it is possible to create a two-dimensional scatter plot to visualize the data. Note that many real-world datasets are highly dimensional and cannot be visualized naively. We will be using `matplotlib` for all data visualization purposes since it offers a wide range of visualization tools and is easy to use.

# In[11]:


plt.plot(X_train, y_train, 'ro', ms=1, mec='k') # the parameters control the size, shape and color of the scatter plot
plt.ylabel('Price in USD')
plt.xlabel('sq.ft')
plt.show()


# ## Bias Trick
# 
# Make sure that `X` takes into consideration the bias $\theta_0$ in the linear model. Hint, recall that the predications of our linear model are of the form:
# 
# $$
# \hat{y} = h_\theta(x) = \theta^T x = \theta_0 + \theta_1 x_1
# $$
# 
# Add columns of ones as the zeroth column of the features (do this for both the training and validation sets).
# 
# ---
# Open `hw1.py` and complete the function `apply_bias_trick`. (5 points)

# In[12]:


from hw1 import apply_bias_trick

X_train = apply_bias_trick(X_train)
X_val = apply_bias_trick(X_val)

X_train.shape, X_val.shape


# In[56]:


X_train


# In[57]:


X_val


# ## Part 2: Single Variable Linear Regression (60 Points)
# Simple linear regression is a linear regression model with a single explanatory varaible and a single target value. 
# 
# $$
# \hat{y} = h_\theta(x) = \theta^T x = \theta_0 + \theta_1 x_1
# $$
# 
# ## Gradient Descent 
# 
# Our task is to find the best possible linear line that explains all the points in our dataset. We start by guessing initial values for the linear regression parameters $\theta$ and updating the values using gradient descent. 
# 
# The objective of linear regression is to minimize the cost function $J$:
# 
# $$
# J(\theta) = \frac{1}{2m} \sum_{i=1}^{n}(h_\theta(x^{(i)})-y^{(i)})^2
# $$
# 
# where the hypothesis (model) $h_\theta(x)$ is given by a **linear** model:
# 
# $$
# h_\theta(x) = \theta^T x = \theta_0 + \theta_1 x_1
# $$
# 
# $\theta_j$ are parameters of your model. and by changing those values accordingly you will be able to lower the cost function $J(\theta)$. One way to accopmlish this is to use gradient descent:
# 
# $$
# \theta_j = \theta_j - \alpha \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}
# $$
# 
# In linear regresion, we know that with each step of gradient descent, the parameters $\theta_j$ get closer to the optimal values that will achieve the lowest cost $J(\theta)$.
# 
# ---
# 
# Open `hw1.py` and complete the function `compute_cost`. (10 points)

# In[13]:


from hw1 import compute_cost
theta = np.array([-1, 2])
J = compute_cost(X_train, y_train, theta)


# In[14]:


J


# Open `hw1.py` and complete the function `gradient_descent`. (30 points)

# In[19]:


from hw1 import gradient_descent

np.random.seed(42)
theta = np.random.random(size=2)
iterations = 40000
alpha = 0.1
theta, J_history = gradient_descent(X_train ,y_train, theta, alpha, iterations)


# You can evaluate the learning process by monitoring the loss as training progress. In the following graph, we visualize the loss as a function of the iterations. This is possible since we are saving the loss value at every iteration in the `J_history` array. This visualization might help you find problems with your code. Notice that since the network converges quickly, we are using logarithmic scale for the number of iterations. 

# In[20]:


plt.plot(np.arange(iterations), J_history)
plt.xscale('log')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss as a function of iterations')
plt.show()


# The pseudo inverse solution is a direct approach to finding the best-fitting parameters of the linear model. In your implementation, **do not use `np.linalg.pinv`**. Instead, use only direct matrix multiplication as you saw in class (you can calculate the inverse of a matrix using `np.linalg.inv`).
# 
# ---
# Open `hw1.py` and complete the function `compute_pinv`. (10 points)

# In[22]:


from hw1 import compute_pinv


# In[23]:


theta_pinv = compute_pinv(X_train ,y_train)
J_pinv = compute_cost(X_train, y_train, theta_pinv)


# We can add the loss value for the theta calculated using the psuedo-inverse to our graph. This is another sanity check as the loss of our model should converge to the psuedo-inverse loss.

# In[24]:


plt.plot(np.arange(len(J_history)), J_history)
plt.xscale('log')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss as a function of iterations')
plt.hlines(y = J_pinv, xmin = 0, xmax = len(J_history), color='r',
           linewidth = 1, linestyle = 'dashed')
plt.show()


# We can use a better approach for the implementation of `gradient_descent`. Instead of performing 40,000 iterations, we wish to stop when the improvement of the loss value is smaller than `1e-8` from one iteration to the next. 
# 
# The learning rate is another factor that determines the performance of our model in terms of speed and accuracy. Complete the function `find_best_alpha`. Make sure you use the training dataset to learn the parameters (thetas) and use those parameters with the validation dataset to compute the cost.
# 
# **After implementing `efficient_gradient_descent` and finding the best alpha value, use them for the rest of the exercise.**
# 
# ---
# Open `hw1.py` and complete the function `efficient_gradient_descent` and `find_best_alpha`. (5 points each)

# In[25]:


from hw1 import efficient_gradient_descent, find_best_alpha


# In[26]:


alpha_dict = find_best_alpha(X_train, y_train, X_val, y_val, 40000)


# We can now obtain the best learning rate from the dictionary `alpha_dict` in a single line.

# In[27]:


best_alpha = min(alpha_dict, key=alpha_dict.get)
print(best_alpha)


# The following code gets the best three alpha values you just calculated and provides a graph with three lines indicating the training loss as a function of iterations. Use it as a test for your implementation. You can change this code as you see fit.

# In[28]:


top_3_alphas = sorted([(value, key) for key, value in alpha_dict.items()], reverse=False)[:3]
top_3_alphas = [x[1] for x in top_3_alphas]

histories = []
for alpha in top_3_alphas:
    params = np.random.random(size=2)
    _, J_history = gradient_descent(X_train ,y_train, params, alpha, num_iters=10000)
    histories.append(J_history)

for i, (alpha, color) in enumerate(zip(top_3_alphas, ['b','g','r'])):
    plt.plot(np.arange(10000), histories[i], color, label='alpha='+str(alpha))

plt.xscale('log')
plt.ylim(0, 0.005)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss as a function of iterations')
plt.legend()
plt.show()


# This is yet another check. This function plots the regression lines of your model and the model based on the pseudoinverse calculation. Both models should exhibit the same trend through the data. 

# In[29]:


plt.figure(figsize=(7, 7))
plt.plot(X_train[:,1], y_train, 'ro', ms=1, mec='k')
plt.ylabel('Price in USD')
plt.xlabel('sq.ft')
plt.plot(X_train[:, 1], np.dot(X_train, theta), 'o')
plt.plot(X_train[:, 1], np.dot(X_train, theta_pinv), '-')

plt.legend(['Training data', 'Linear regression', 'Best theta']);


# ## Part 2: Multivariate Linear Regression
# 
# In most cases, you will deal with datasets that have more than one feature. It can be as little as two features and up to thousands of features. In those cases, we use a multivariate linear regression model. The regression equation is almost the same as the simple linear regression equation:
# 
# $$
# \hat{y} = h_\theta(\vec{x}) = \theta^T \vec{x} = \theta_0 + \theta_1 x_1 + ... + \theta_n x_n
# $$
# 
# 
# If you wrote proper vectorized code, this part should be trivial and work without changes. If this is not the case, you should go back and edit your functions such that they support both multivariate and single variable regression. **Your code should not check the dimensionality of the input before running**.

# In[43]:


# Read comma separated data
df = pd.read_csv('data.csv')
df.head()


# ## Preprocessing
# 
# Like in the single variable case, we need to create a numpy array from the dataframe. Before doing so, we should notice that some of the features are clearly irrelevant so we will go ahead and drop them.

# In[44]:


X = df.drop(columns=['price', 'id', 'date']).values
y = df['price'].values


# Use the same `preprocess` function you implemented previously. Notice that proper vectorized implementation should work regardless of the dimensionality of the input. You might want to check that your code in the previous parts still works.

# In[45]:


# preprocessing
X, y = preprocess(X, y)


# In[46]:


# training and validation split 
np.random.seed(42)
indices = np.random.permutation(X.shape[0])
idx_train, idx_val = indices[:int(0.8*X.shape[0])], indices[int(0.8*X.shape[0]):]
X_train, X_val = X[idx_train,:], X[idx_val,:]
y_train, y_val = y[idx_train], y[idx_val]


# Using 3D visualization, we can still observe trends in the data. Visualizing additional dimensions requires advanced techniques we will learn later in the course.

# In[47]:


get_ipython().run_line_magic('matplotlib', 'inline')
import mpl_toolkits.mplot3d.axes3d as p3
fig = plt.figure(figsize=(5,5))
ax = p3.Axes3D(fig)
xx = X_train[:, 1][:1000]
yy = X_train[:, 2][:1000]
zz = y_train[:1000]
ax.scatter(xx, yy, zz, marker='o')
ax.set_xlabel('bathrooms')
ax.set_ylabel('sqft_living')
ax.set_zlabel('price')
plt.show()


# Make sure the functions `apply_bias_trick`, `compute_cost`, `gradient_descent`, `efficient_gradient_descent` and `compute_pinv` work on the multi-dimensional dataset. If you make any changes, make sure your code still works on the single variable regression model. 

# In[48]:


# bias trick
X_train = apply_bias_trick(X_train)
X_val = apply_bias_trick(X_val)

X_train.shape, X_val.shape


# In[49]:


# calculating the cost
shape = X_train.shape[1]
theta = np.ones(shape)
J = compute_cost(X_train, y_train, theta)


# In[39]:


# running the efficient version of gradient descent
np.random.seed(42)
shape = X_train.shape[1]
theta = np.random.random(shape)
iterations = 40000
theta, J_history = efficient_gradient_descent(X_train ,y_train, theta, best_alpha, iterations)


# In[ ]:


# calculating the pseudoinverse
theta_pinv = compute_pinv(X_train ,y_train)
J_pinv = compute_cost(X_train, y_train, theta_pinv)


# We can use visualization to make sure the code works well. Notice we use logarithmic scale for the number of iterations, since gradient descent converges after ~500 iterations.

# In[ ]:


plt.plot(np.arange(len(J_history)), J_history)
plt.xscale('log')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss as a function of iterations - multivariate linear regression')
plt.hlines(y = J_pinv, xmin = 0, xmax = len(J_history), color='r',
           linewidth = 1, linestyle = 'dashed')
plt.show()


# ## Part 3: Forward feature selection (15 points)
# 
# Adding additional features to our regression model makes it more complicated but does not necessarily improves performance. 
# 
# Forward feature selection is a greedy, iterative algorithm used to select the most relevant features for a predictive model. The objective of this algorithm is to improve the model's performance by identifying and using only the most relevant features, potentially reducing overfitting, improving accuracy, and reducing computational cost. 
# 
# Implement forward feature selection using the following guidelines: 
# 1. Start with an empty set of selected features.
# 1. For each feature not yet in the selected set, do the following:
#     1. Add the feature to the selected set temporarily.
#     1. Train a model using the current set of selected features and evaluate its performance by calculating the cost or error on a validation set.
#     1. Remove the temporarily added feature from the selected set.
# 1. Choose the feature that resulted in the best model performance and permanently add it to the selected set.
# 1. Repeat steps 2-3 until you have 5 features (not including the bias).
# 
# ---
# Open `hw1.py` and complete the function `forward_feature_selection`.
# 
# **Note that you should use the inputs as provided in the next cell and don't forget to use the bias trick inside `forward_feature_selection`**.

# In[ ]:


df = pd.read_csv('data.csv')
feature_names = df.drop(columns=['price', 'id', 'date']).columns.values
X = df.drop(columns=['price', 'id', 'date']).values
y = df['price'].values

# preprocessing
X, y = preprocess(X, y)

# training and validation split
np.random.seed(42)
indices = np.random.permutation(X.shape[0])
idx_train, idx_val = indices[:int(0.8*X.shape[0])], indices[int(0.8*X.shape[0]):]
X_train, X_val = X[idx_train,:], X[idx_val,:]
y_train, y_val = y[idx_train], y[idx_val]


# In[ ]:


from hw1 import forward_feature_selection


# In[ ]:


ffs = forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations)
for feature in ffs:
    print(feature_names[feature])


# ## Part 4: Polynomial regression (15 points)
# 
# Implement a function to create polynomial features up to a degree of 2 for given dataset. The function should take a pandas DataFrame as input and should return a new DataFrame with all possible square features, including the original features. When you create the new dataframe, make sure the feature names also represent the transformation performed. For example: `sqft_lot`, `yr_built^2`, `bedrooms*bathrooms`, ...
# 
# After you obtain the polynomial dataframe, use forward feature selection and print the 5 best features.
# 
# Open `hw1.py` and complete the function `create_square_features`.

# In[ ]:


from hw1 import create_square_features


# In[ ]:


df = pd.read_csv('data.csv')

y = df['price'].values
df = df.drop(columns=['price', 'id', 'date'])
df = df.astype('float64')
df_poly = create_square_features(df)
X = df_poly.values
X.shape, y.shape


# In[ ]:


# preprocessing
X, y = preprocess(X, y)


# In[ ]:


# training and validation split
np.random.seed(42)
indices = np.random.permutation(X.shape[0])
idx_train, idx_val = indices[:int(0.8*X.shape[0])], indices[int(0.8*X.shape[0]):]
X_train, X_val = X[idx_train,:], X[idx_val,:]
y_train, y_val = y[idx_train], y[idx_val]


# In[ ]:


X_train.shape, X_val.shape, y_train.shape, y_val.shape


# In[ ]:


feature_names = df_poly.columns.values


# In[ ]:


ffs = forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations)


# In[ ]:


for feature in ffs:
    print(feature_names[feature])

