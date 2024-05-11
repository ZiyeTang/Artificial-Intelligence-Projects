'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
import numpy as np

def k_nearest_neighbors(image, train_images, train_labels, k):
    '''
    Parameters:
    image - one image
    train_images - a list of N images
    train_labels - a list of N labels corresponding to the N images
    k - the number of neighbors to return

    Output:
    neighbors - 1-D array of k images, the k nearest neighbors of image
    labels - 1-D array of k labels corresponding to the k images
    '''

    distance = (np.sum((train_images - image)**2, axis = 1))**0.5
    idx_order = np.argsort(distance)
    neighbors = []
    labels = []

    for i in range(min(k, len(idx_order))):
        neighbors.append(train_images[idx_order[i]])
        labels.append(train_labels[idx_order[i]])

    return np.array(neighbors), np.array(labels)


def classify_devset(dev_images, train_images, train_labels, k):
    '''
    Parameters:
    dev_images (list) -M images
    train_images (list) -N images
    train_labels (list) -N labels corresponding to the N images
    k (int) - the number of neighbors to use for each dev image

    Output:
    hypotheses (list) -one majority-vote labels for each of the M dev images
    scores (list) -number of nearest neighbors that voted for the majority class of each dev image
    '''
    hypotheses = []
    scores = []
    for img in dev_images:
        _, lbs = k_nearest_neighbors(img, train_images, train_labels, k)
        hypotheses.append((sum(lbs) > k//2) + 0)
        scores.append(max(sum(lbs), k - sum(lbs)))


    return hypotheses, scores


def confusion_matrix(hypotheses, references):
    '''
    Parameters:
    hypotheses (list) - a list of M labels output by the classifier
    references (list) - a list of the M correct labels

    Output:
    confusions (list of lists, or 2d array) - confusions[m][n] is 
    the number of times reference class m was classified as
    hypothesis class n.
    accuracy (float) - the computed accuracy
    f1(float) - the computed f1 score from the matrix
    '''
    
    confusions = np.zeros((2,2))

    for i in range(len(hypotheses)):
        confusions[references[i]-0][hypotheses[i]-0] += 1

    accuracy = (confusions[1,1] + confusions[0,0])/np.sum(confusions)
    Precission = confusions[1][1]/sum(confusions[:,1])
    Recall = confusions[1][1]/sum(confusions[1])
    f1 = 2/(1/Precission + 1/Recall)
    return confusions, accuracy, f1