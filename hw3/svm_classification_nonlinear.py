from mnist import MNIST
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def plot_image(image_list):
    image = np.asarray(image_list)

    plt.figure()
    plt.imshow(np.reshape(image, (28,28)), cmap='gray_r')
    plt.show()


def perturb_data(images_test):
    print("\nIntroducing Disorder to Testing Set...")
    


'''Here, clf can be the SVM object returned by
the SVC library (i.e., the fit function). You
can directly call predict on that object.'''
def test(clf, images, labels):
    print("Predicting...")
    pred = clf.predict(images)
    #compare with labels
    print(pred.shape)
    print("Calculating Accuracy")
    acc = np.sum(np.where(pred == labels, 1, 0))/images.shape[0]
    return acc


def train(images, labels):
    print("Training...")
    clf = make_pipeline(StandardScaler(), svm.SVC(gamma='auto', max_iter = 1000))
    clf.fit(images, labels)
    return clf


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

    #train
    clf = train(images_train, labels_train)

    #test
    acc_train = test(clf, images_train, images_train)
    print(f"Training set accuracy: {acc_train:.6f}")
    acc_test = test(clf, images_test, labels_test)    
    print(f"Testing set accuracy: {acc_test:.6f}")

    #introduce disorder
    perturb_data(images_test)


    

