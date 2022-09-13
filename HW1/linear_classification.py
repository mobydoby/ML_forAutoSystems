from mnist import MNIST
import matplotlib.pyplot as plt
import numpy as np
import sys

def plot_image(image_list):
    image = np.asarray(image_list)

    plt.figure()
    plt.imshow(np.reshape(image, (28,28)), cmap='gray_r')
    plt.show()

"""
Inputs: all_images = N x 784 matrix
        all_labels = Nx1 - column vector
Outputs: W (weights): 785x1 column vector
Effects: 
    Uses Linear Regression to train a binary classifier for label1 and label2
    The input images only for labels 1 and 2 are used for the classifier. 
    W = (XX.T)^(-1)Xy
    The pseudo inverse function can be modified to use more sophisticated data
    manipulation or the SVD. 

    See pseudoi() function
"""
def train(all_images, all_labels, label1: int, label2: int)->np.ndarray:
    print(f"Training weights for {label1} and {label2}\n.\n.\n.")

    #creates a mask for the features true if the label is either l1 or l2
    valid_indices = (all_labels != label1)*(all_labels != label2)

    #deletes all rows in features and labels that are not the correct label (1 or 2)
    features = np.delete(all_images, valid_indices, 0)
    y = np.delete(all_labels, valid_indices, 0)

    #change label1 to 0 and label2 to 1
    y = np.where(y == label1, 0, 1)
    
    #append a 1 to features (for bias) at the END
    # [[x1 1
    #   x2 1
    #   x3 1]]
    bias = np.expand_dims(np.ones(features.shape[0]), axis = 0).T
    X = np.append(features, bias, axis = 1).T
    XX_T = X.dot(X.T)
    Xy = X.dot(y)
    pinv = pseudoi(XX_T)
    W = pinv.dot(Xy)
    return W

"""
returns the pseudo-inverse of arr
"""
def pseudoi(arr: np.array, epsilon = 0.0001)->np.array:
    return np.linalg.inv(arr + np.identity(arr.shape[0])*epsilon)

def get_optimal_thresh(images_train, labels_train, w):
    return 0.5

"""
Inputs: all_images_test: Nx784 matrix with all test images
        all_labels_test: Nx1 vector of labels
        w is 785x1 matrix of weights
        label1 and label2 are the labels (0-9) being compared
        thresh: threshold for classfication
Output: outputs a ratio of correct/total labels
"""
def test(all_images_test, all_labels_test, label1, label2, w, thresh)->float:
    """
        - creates a mask for the features true if the label is either l1 or l2
        - deletes all rows in features and labels that are not the correct label (1 or 2)
    """
    valid_indices = (all_labels_test != label1)*(all_labels_test != label2)
    features = np.delete(all_images_test, valid_indices, 0)
    y = np.delete(all_labels_test, valid_indices, 0)
    #change label1 to 0 and label2 to 1
    y = np.where(y == label1, 0, 1)
    
    #append a 1 to features (for bias) at the END
    bias = np.expand_dims(np.ones(features.shape[0]), axis = 0).T
    X = np.append(features, bias, axis = 1)

    #predict with weights
    pred = X.dot(w)
    pred = np.where(pred>thresh, 0, 1)

    #compare with labels 
    # print(y.shape[0], type(y.shape[0]))
    ratio_correct = np.sum(np.where(pred == y, 0, 1))/y.shape[0] #y.shape[0] = N (number of images)
    print(f"Training for {label1} and {label2} done.\nPercentage Correct = {ratio_correct:.6f}\n=============================")


if __name__ == "__main__":

    # load the data
    mndata = MNIST('datasets/MNIST/raw')
    images_list, labels_list = mndata.load_training()
    images_list_test, labels_list_test = mndata.load_testing()
    
    # training 
    images_list = np.array(images_list)
    labels_list = np.array(labels_list)
    images_list_test = np.array(images_list_test)
    labels_list_test = np.array(labels_list_test)
    print(labels_list_test.shape)

    val, counts = np.unique(labels_list, return_counts=True)
    label_count = dict(zip(val, counts))
    print(label_count)

    # testing
    # W = train(images_list, labels_list, 0, 1, 0.0001)
    # test(images_list_test, labels_list_test, 0, 1, W, 0.5)
   
    """
    for every pair of numbers from 0-9, find the classifier and determine accuracy on test data
    """
    for i in range(10):
        for j in range(i+1, 10):
            W = train(images_list, labels_list, i, j)
            thresh = get_optimal_thresh(images_list, labels_list, W)
            test(images_list_test, labels_list_test, i, j, W, thresh)

