from mnist import MNIST
import matplotlib.pyplot as plt
import numpy as np
from decision_tree import DecisionTree
import pandas as pd


def plot_image(image_list):
    image = np.asarray(image_list)

    plt.figure()
    plt.imshow(np.reshape(image, (28,28)), cmap='gray_r')
    plt.show()

if __name__ == "__main__":
    
    mndata = MNIST('../datasets/MNIST/raw')
    images_list, labels_list = mndata.load_training()
    images_list_test, labels_list_test = mndata.load_testing()

    images_train = np.asarray(images_list).astype(float)
    labels_train = np.asarray(labels_list).astype(float)

    num_leaves = 10

    # importing weights
    data = pd.read_csv("../HW1/weights.csv").to_numpy()
