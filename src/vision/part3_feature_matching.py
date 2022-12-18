#!/usr/bin/python3

import numpy as np

from typing import Tuple


def compute_feature_distances(
    features1: np.ndarray,
    features2: np.ndarray
) -> np.ndarray:
    """
    This function computes a list of distances from every feature in one array
    to every feature in another.

    Using Numpy broadcasting is required to keep memory requirements low.

    Note: Using a double for-loop is going to be too slow. One for-loop is the
    maximum possible. Vectorization is needed.
    See numpy broadcasting details here:
        https://cs231n.github.io/python-numpy-tutorial/#broadcasting

    Args:
        features1: A numpy array of shape (n1,feat_dim) representing one set of
            features, where feat_dim denotes the feature dimensionality
        features2: A numpy array of shape (n2,feat_dim) representing a second
            set of features (n1 not necessarily equal to n2)

    Returns:
        dists: A numpy array of shape (n1,n2) which holds the distances (in
            feature space) from each feature in features1 to each feature in
            features2
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    n1, feat_dim = features1.shape
    n2, _ = features2.shape
    dists = np.zeros([n1, n2])
    for i in range(feat_dim):
        dists += (features1[:, i].reshape(n1, 1) - features2[:, i].reshape(1, n2)) ** 2
    dists =np.sqrt(dists)

    '''
    dists=np.ones((features1.shape[0],features2.shape[0],features1.shape[1]))
    for i in range(features2.shape[0]):
        dists[:,i]=features1
    for i in range(features1.shape[0]):
        dists[i]=dists[i]-features2
    dists=dists*dists
    dists=np.sum(dists,axis=2)
    dists=np.sqrt(dists)
    '''

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dists


def match_features_ratio_test(
    features1: np.ndarray,
    features2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """ Nearest-neighbor distance ratio feature matching.

    This function does not need to be symmetric (e.g. it can produce different
    numbers of matches depending on the order of the arguments).

    To start with, simply implement the "ratio test", equation 7.18 in section
    7.1.3 of Szeliski. There are a lot of repetitive features in these images,
    and all of their descriptors will look similar. The ratio test helps us
    resolve this issue (also see Figure 11 of David Lowe's IJCV paper).

    You should call `compute_feature_distances()` in this function, and then
    process the output.

    Args:
        features1: A numpy array of shape (n1,feat_dim) representing one set of
            features, where feat_dim denotes the feature dimensionality
        features2: A numpy array of shape (n2,feat_dim) representing a second
            set of features (n1 not necessarily equal to n2)

    Returns:
        matches: A numpy array of shape (k,2), where k is the number of matches.
            The first column is an index in features1, and the second column is
            an index in features2
        confidences: A numpy array of shape (k,) with the real valued confidence
            for every match

    'matches' and 'confidences' can be empty, e.g., (0x2) and (0x1)
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    dists=compute_feature_distances(features1,features2)
    #X_r=np.zeros(features1.shape[0])
    m=[]
    c=[]
    #Y_r=np.zeros(features2.shape[0])
    cutoff=0.8
    for i in range(features1.shape[0]):
        line=np.copy(dists[i])
        line.sort()
        if line[0]/line[1]<cutoff:
            c.append(line[0]/line[1])
            m.append([i,np.where(dists[i]==line[0])[0][0]])
    #for i in range(features2.shape[0]):
    #    line=np.copy(dists[:,i])
    #    line.sort()
    #    if line[0] / line[1] > curoff:
    #        Y_r[i]=line[0]/line[1]
    if m==[]:
        matches=np.zeros((0,2))
        confidences=np.zeros((0,1))
    else:
        matches=np.array(m)
        confidences=np.array(c)


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return matches, confidences
