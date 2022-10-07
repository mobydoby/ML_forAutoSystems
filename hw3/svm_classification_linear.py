from mnist import MNIST
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from scipy import stats
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def plot_image(image_list):
    image = np.asarray(image_list)

    plt.figure()
    plt.imshow(np.reshape(image, (28,28)), cmap='gray_r')
    plt.show()


'''clf can be an object (e.g., a dictionary)
that can be used to store the weights, 
offsets and whatever other information you need'''
def test(all_planes, images, labels):

    """
    Inputs: all_planes = a list of all SVCs
            images = N x 784
            labels = N x 1
    Outputs: N x 1 predictions
    Effects: For each classifier (plane), classify the image set. 
             This will result in N x 45 vector of predictions (N images with 45 planes)
             Calculate the mode in each row (most common class for each feature)
    """
    all_pred = np.empty((images.shape[0], 0), dtype=int)
    for i in range(len(all_planes)):
        print(f"predicting for {i}")
        # get predictor
        predictor = all_planes[i]
        pred = np.expand_dims(predictor.predict(images), axis=1) # N x 1
        
        #add prediction for this plane to all_preds
        all_pred = np.concatenate((all_pred, pred), axis = 1)

    #calculate the mode in each row, that is the actual prediction
    pred_Y = stats.mode(all_pred, keepdims=True, axis=1)[0]
    print(pred_Y, pred_Y.shape)
    print(labels.shape)

    #compare with labels
    print("calculating loss")
    mask = np.where(pred_Y.T == labels.T, 1, 0)
    print(mask.shape)
    accuracy = np.sum(mask)/labels.shape[0]
    print(accuracy)
    return accuracy

def train(images:np.array, labels, label1:int, label2:int):
    """
    Inputs: Images: 60000 x 784
            labels: 60000 x 1
            label1 and label2 are ints
    Outputs: Vector of Weights, Hyperplane in 785 dimensions (785 x 1)
    """
    
    print(f"Training weights for {label1} and {label2}\n...")

    #creates a mask for the features true if the label is either l1 or l2
    valid_indices = (labels != label1)*(labels != label2)

    #deletes all rows in features and labels that are not the correct label (1 or 2)
    features = np.delete(images, valid_indices, 0)
    y = np.delete(labels, valid_indices, 0)

    plane = make_pipeline(StandardScaler(), SVC(kernel = "linear", max_iter = 100000, C=1))
    plane.fit(features, y)

    print(plane)
    return plane

if __name__ == "__main__":
    
    mndata = MNIST('../datasets/MNIST/raw')

    images_list, labels_list = mndata.load_training()
    images_list_test, labels_list_test = mndata.load_testing()

    images_train = np.asarray(images_list).astype(float)
    labels_train = np.asarray(labels_list).astype(float)
    images_test = np.asarray(images_list_test).astype(float)
    labels_test = np.asarray(labels_list_test).astype(float)

    images_train = np.hstack((images_train, np.ones((images_train.shape[0], 1))))
    images_test = np.hstack((images_test, np.ones((images_test.shape[0], 1))))

    # print(images_train.shape) #(60000, 784)
    # print(labels_train.shape) #(60000,)
    # print(images_test.shape)  #(10000, 784)
    # print(labels_test.shape)  #(10000,)

    # hyperplanes is a dictionary of (int, int): Weight
    hyperplanes = []
    for i in range(10):
        for j in range(i+1, 10):
            plane=train(images_train, labels_train, i, j)
            hyperplanes.append(plane)

    train_acc = test(hyperplanes, images_train, labels_train)
    test_acc = test(hyperplanes, images_test, labels_test)
    print()


        
    


