from mnist import MNIST
import matplotlib.pyplot as plt
import numpy as np
from decision_tree import DecisionTree
import pandas as pd
import pickle
import sys
from os import path


def plot_image(image_list):
    image = np.asarray(image_list)

    plt.figure()
    plt.imshow(np.reshape(image, (28,28)), cmap='gray_r')
    plt.show()

def get_features_from_images(weight: np.array, images_train: np.array, images_test: np.array):
    """
    Returns the weights for the tree. appends the bias term to training data and take weights.dot(images)
    Inputs: Weight vector (45 x 785)
            images_train (N x 784)
            images_test (N x 784)
    Outputs: train_features (45 x N), test_features (45 x N) 
    """
    #adding bias to training
    bias_train = np.expand_dims(np.ones(images_train.shape[0]), axis = 0).T
    images_train = np.append(images_train, bias_train, axis = 1)

    #adding bias to training
    bias_test = np.expand_dims(np.ones(images_test.shape[0]), axis = 0).T
    images_test = np.append(images_test, bias_test, axis = 1)

    return weight@images_train.T, weight@images_test.T

if __name__ == "__main__":
    if len(sys.argv) != 2 and len(sys.argv) != 3 and (sys.argv[1] not in ['train', 'test']):
        raise RuntimeError("Usage: decision_tree_classification.py <'train' or 'test'> <if 'test': saved_model_filepath>")
    
    # Load Mnist data
    mndata = MNIST('../datasets/MNIST/raw')
    images_list, labels_list = mndata.load_training()
    images_list_test, labels_list_test = mndata.load_testing()

    images_train = np.asarray(images_list).astype(float)
    labels_train = np.asarray(labels_list).astype(float)
    images_test = np.asarray(images_list_test).astype(float)
    labels_test = np.asarray(labels_list_test).astype(float)

    num_leaves = 10

    # importing weights
    W = pd.read_csv("../HW1/weights.csv").to_numpy()[0:,1:] # W.shape = 45 x N
    print(W.shape)

    X_train, X_test = get_features_from_images(W, images_train, images_test)

    # print(X_train.shape, X_test.shape) #(45, 60000), (45, 10000)
    # print(labels_train.shape, labels_test.shape) #(60000,) (10000,)
   
    if sys.argv[1] == 'test':
        #test but no file path given
        if len(sys.argv) == 2 or path.exists(sys.argv[2]) == False:
            raise RuntimeError("No model filepath found")
        with open(sys.argv[2], "rb") as input_file:
            Model = pickle.load(input_file)
        input_file.close()
    else:
        #train model
        Model = DecisionTree()
        Model.train(X_train, labels_train, 100)

        #if path given for saving data, save the thing.
        if len(sys.argv) == 3: 
            with open(sys.argv[2], "wb") as out_file:
                pickle.dump(Model, out_file)
            out_file.close()

    #test model
    print(Model.show_labels())
    Model.test(X_test, labels_test)



