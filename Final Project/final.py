import numpy as np
import deeplake
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from sklearn.utils import resample,shuffle
import matplotlib.pyplot as plt
import time

path_prepend = 'Final Project/'

num_copies = 100
batch_size = 1000

# Load Function
def load(f):
    return np.load(f, allow_pickle=True)['arr_0']

def reshape_and_resample(x_train, x_test, y_train, y_test):
  # the following code was adapted from https://towardsdatascience.com/using-fastai-to-classify-japanese-kanji-characters-47d7edd4d569
    # it resamples the data to make sure that every character appears a similar number of times
    # otherwise the dataset will be unbalanced and that causes problems

    # Reshape the 3D ndarray (69804 x 64 x 64) into a 2D array (69804 x 4096) 
    new_x_train = x_train
    dim_0, dim_1, dim_2 = new_x_train.shape
    new_x_train = np.array(new_x_train).reshape(dim_0, dim_1 * dim_2)

    # Add a label column
    label = pd.DataFrame(y_train)
    new_x_train = pd.DataFrame(new_x_train)
    new_x_train['label'] = label

    # Add a count column (number of occurences of a label)
    # New shape is (69804 x 4098)
    count = pd.DataFrame(new_x_train['label'].value_counts())
    count.columns = ['count']
    count['label'] = count.index
    new_x_train = pd.merge(new_x_train,count, on='label')

    # Array for already processed labels
    my_input = []

    # Sample Size
    sample_size = 10

    # DataFrame for remaining rows
    df_rest = new_x_train.iloc[:1]

    # DataFrame for resampled rows
    df_rest_upsampled = new_x_train.iloc[:1]

    # Boolean if rows should be added to the test dataset 2
    restAdapt = False 

    for item in range(len(new_x_train)):
        if (new_x_train.iloc[item,4096] not in my_input): #Check if the label has already been processed
            # Selction of label and DataFrame
            df_1 = new_x_train[new_x_train['label'] == new_x_train.iloc[item,4096]]

            # Check if count bigger than sample_size, then randomly select sample size rows
            if df_1.iloc[0,4097] > sample_size:
                testSeriesNP = df_1.index.to_numpy() # indexes of the labels
                training, testing, _, _ = train_test_split(testSeriesNP,testSeriesNP, test_size=sample_size, random_state=42) # randomly select the sample_size
                df_1_upsampled = new_x_train.iloc[testing] # randomly selected sample size
                df_rest_upsampled = new_x_train.iloc[training] # remaining rows, used for test dataset 2
                restAdapt = True
                my_input.append(new_x_train.iloc[item,4096])

            else: # if count is smaller than sample_size, resample to sample_size
                df_1_upsampled = resample(df_1,random_state=42,n_samples=sample_size,replace=True)
                my_input.append(new_x_train.iloc[item,4096])
                
            # Create the final DataFrame
            if item == 0:
                df_upsampled = df_1_upsampled
                df_rest = df_rest_upsampled
                restAdapt = False
            else:
                df_upsampled = pd.concat([df_1_upsampled,df_upsampled])
                if restAdapt:
                    df_rest = pd.concat([df_rest_upsampled,df_rest])
                    restAdapt = False

    # Delete
    df_upsampled.drop(['count'], axis=1, inplace=True)

    # Delete not required rows and columns
    df_rest.drop(df_rest.tail(1).index,inplace=True)
    df_rest.drop(['count'], axis=1, inplace=True)

    # Extract label
    new_y_train = df_upsampled['label']
    new_y_train = new_y_train.to_numpy().astype(int)

    # Delete column
    df_upsampled.drop(['label'], axis=1, inplace=True)  

    # Dimensions
    new_dim1 = df_upsampled.shape[0]
    old_dim = 64

    # reshape into the new dimension
    new_x_train = np.array(df_upsampled).reshape(new_dim1, old_dim,old_dim)
        
    # Extract label
    new_y_test2 = df_rest['label']
    new_y_test2 = new_y_test2.to_numpy().astype(int)

    # Delete column
    df_rest.drop(['label'], axis=1, inplace=True)

    # concatenate test labels and new labels for the final test dataset
    final_y_test = np.concatenate([y_test,new_y_test2], -1)   

    # Dimensions - original test dataset
    dim_0, dim_1, dim_2 = x_test.shape

    # Reshaping for Pandas DataFrame
    new_x_test = np.array(x_test).reshape(dim_0, dim_1 * dim_2)
    new_x_test = pd.DataFrame(new_x_test)

    # Concatenate test and rest
    final_x_test = pd.concat([new_x_test,df_rest])    

    # Dimensions
    new_dim1 = final_x_test.shape[0]
    old_dim = 64

    # reshape into the new dimension for Numpy ndarray 
    final_x_test = np.array(final_x_test).reshape(new_dim1, old_dim,old_dim)

    return (new_x_train, new_y_train, final_x_test, final_y_test)

def sigmoid(X, W):
    return 1 / (1 + np.exp(-1 * X.dot(W)))

def sigmoid_idx_batch(X_idx, X, W, batch_size):
    ones = np.array([1] * batch_size)
    sigmoids = []
    for offset in range(0, int(np.ceil(X_idx.shape[0] / batch_size)) ):
        base = offset * batch_size
        print(f"idx = {base}, length = {X_idx.shape[0]}, {100 * base / X_idx.shape[0]}%")
        end = min(base + batch_size, X_idx.shape[0])
        arr = np.concatenate((ones.reshape((batch_size, 1)), 
                           X[X_idx[base:end, 0]],
                           X[X_idx[base:end, 1]]),
                           axis=1)
        sigmoids.append(sigmoid(arr, W))
    return np.concatenate(sigmoids)

def compute_loss(X, y, W):
    y_hat = sigmoid(X, W)
    return -1 * sum([y[idx] * np.log(y_hat[idx]) + (1 - y[idx]) * (np.log(1 - y_hat[idx])) for idx in range(len(y))]) / len(y)

def delta_weights(X, y, W):
    p = sigmoid(X, W)
    Dw = X.T.dot(p - y) / len(y) 
    return Dw


def gradient_descent(X, y, W, learning_rate, iterations):
    loss_per = []
    for iter in range(iterations):
        deltas = delta_weights(X, y, W)
        W -= learning_rate * deltas
        loss = compute_loss(X, y, W)
        loss_per.append([iter, loss])
    return (W, np.array(loss_per))


def time_sig_batch(X_train, X_train_indices, batch_size):
    start = time.time()
    ones = np.array([1] * batch_size)
    res = (np.concatenate((ones.reshape((batch_size, 1)), 
                           X_train[X_train_indices[:batch_size, 0]],
                           X_train[X_train_indices[:batch_size, 1]]),
                           axis=1)).dot(W)
    end = time.time()
    #print(f"Time to compute {batch_size} sigmoid(s): {end - start}")
    #print(f"Time to compute all sigmoids with batch size {batch_size}: {(end - start) * (X_train_indices.shape[0] / batch_size)}")
    return (end - start) * (X_train_indices.shape[0] / batch_size)

X_train = load(path_prepend + 'dataset/k49-train-imgs.npz')
y_train = load(path_prepend + 'dataset/k49-train-labels.npz')

print("Loaded training images and labels")

# This is a binary classifier, so we need two sets of data
# This means that we need to transform the data a bit
# first, we need to reshape the 3D tensor into a 2d array

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1] * X_train.shape[2]))

class_df = pd.DataFrame(y_train,columns=['class'])
class_df['x_index'] = class_df.index
class_df = class_df.groupby('class').head(num_copies)
value_counts = class_df['class'].value_counts(dropna=True, sort=True)

values = pd.DataFrame(value_counts)
values = values.reset_index()
values.columns = ['class', 'occurence'] # change column names

df = pd.merge(class_df,values,on ='class')
print(df)

X_train = X_train[df['x_index']]
y_train = y_train[df['x_index']]

# Next, we need to combine images together
# We want to pair up the images before we train the model, so that all the array operations are very fast
# We'll start with the naive way to do this, combining every image with every image (including self matches)
# We can speed this up by only working with the indexes
X_train_idx = np.array([x for x in range(X_train.shape[0])])
assert np.array_equal(X_train, X_train[X_train_idx])

# Now we can make a copy of this array len times
print(X_train_idx)
X_train_all_copies = np.hstack([X_train_idx] * X_train.shape[0])
print(X_train_all_copies.shape)
X_train_pair_shifted = np.copy(X_train_idx)
X_train_shifting = np.copy(X_train_idx)
for idx in range(1, X_train.shape[0]):
    print(f"{X_train_pair_shifted.shape}, {X_train_shifting.shape}, {100 * idx /  X_train.shape[0]}%")

    X_train_shifting = np.hstack([X_train_shifting, X_train_shifting[0]])
    X_train_shifting = np.delete(X_train_shifting, 0)
    X_train_pair_shifted = np.hstack([X_train_pair_shifted, np.copy(X_train_shifting)])

print(X_train_pair_shifted.shape)
assert X_train_pair_shifted.shape == X_train_all_copies.shape

#for idx in range(X_train_all_copies.shape[0]):
#    print(idx, X_train_all_copies[idx], X_train_pair_shifted[idx])
#    print(y_train[X_train_all_copies[idx]], 
#          y_train[X_train_pair_shifted[idx]])

# generate labels
Y_equality_label = np.array([y_train[X_train_all_copies[idx]] == y_train[X_train_pair_shifted[idx]]
                             for idx in range(X_train_all_copies.shape[0])])

print(np.count_nonzero(Y_equality_label.astype(int)))

# the code is just a simple logistic regression
# we feed it the pixel values of two images, and it tries to figure out if the images are of the same thing
# now we zip the indices together
X_train_indices = np.concatenate((X_train_all_copies.reshape((X_train_all_copies.shape[0], 1)), 
                                  X_train_pair_shifted.reshape((X_train_pair_shifted.shape[0], 1))
                                  ), axis=1)
print(X_train_indices)

W = np.array([0.0 for _ in range(len(X_train[0, :]) * 2 + 1)])

'''
batch_sizes = [1, 100, 1000, 2000, 3000, 5000, 10000, 100000]
avg_size = 500
for size in batch_sizes:
    avg = 0
    for _ in range(avg_size):
        avg += time_sig_batch(X_train, X_train_indices, size)
    print(f"Average total time estimate for batch size {size}: {avg / avg_size}")
'''
start = time.time()
sigs = sigmoid_idx_batch(X_train_indices, X_train, W, batch_size)
end = time.time()
print(f"Time to compute one run: {end - start}")

learning_rate = 0.01
iterations = 1000

(W, loss_per) = gradient_descent(X_train, y_train, W, learning_rate, iterations)

plt.plot(loss_per[:, 0], loss_per[:, 1], label="Log Loss", color="blue")
plt.xlabel("Iteration")
plt.ylabel("Log Loss")
plt.title("Logistic Regression Training")
plt.legend()
plt.show()