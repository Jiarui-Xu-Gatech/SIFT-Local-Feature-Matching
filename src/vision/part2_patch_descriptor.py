#!/usr/bin/python3

import numpy as np


def compute_normalized_patch_descriptors(
    image_bw: np.ndarray, X: np.ndarray, Y: np.ndarray, feature_width: int
) -> np.ndarray:
    """Create local features using normalized patches.

    Normalize image intensities in a local window centered at keypoint to a
    feature vector with unit norm. This local feature is simple to code and
    works OK.

    Choose the top-left option of the 4 possible choices for center of a square
    window.

    Args:
        image_bw: array of shape (M,N) representing grayscale image
        X: array of shape (K,) representing x-coordinate of keypoints
        Y: array of shape (K,) representing y-coordinate of keypoints
        feature_width: size of the square window

    Returns:
        fvs: array of shape (K,D) representing feature descriptors
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    fvs=np.ones((X.shape[0],feature_width*feature_width))
    for i in range(X.shape[0]):
        vector=image_bw[int(Y[i]-int(feature_width/2)+1):int(Y[i]+int(feature_width/2)+1),int(X[i]-int(feature_width/2)+1):int(X[i]+int(feature_width/2)+1)].flatten()
        vector=vector/np.linalg.norm(vector)
        fvs[i]=vector


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return fvs
