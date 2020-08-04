# This code is based on: https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py

"""Calculates the Frechet Inception Distance (FID) to evalulate generative models.
The FID metric calculates the distance (in a perceptual feature space) between two distributions of samples."""

import numpy as np
from scipy import linalg


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1) &
    X_2 ~ N(mu_2, C_2) is d^2 = ||mu_1-mu_2||^2 + Tr(C_1+C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array with sample mean of embeddings of generated samples.
    -- mu2   : Numpy array with sample mean of embeddings of test set samples.
    -- sigma1: Numpy array with covariance matrix of embeddings of generated samples.
    -- sigma2: Numpy array with covariance matrix of embeddings of test set samples.
    Returns:
    -- fd    : The Frechet Distance."""

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape==mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape==sigma2.shape, 'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    # Calculate and return the Frechet Distance
    tr_covmean = np.trace(covmean)
    fd = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2*tr_covmean
    return fd


def calculate_fid_from_embedding(eval_data, ref_data):
    """Calculates the FID for provided embeddings of the generated data ('eval_data') & of the test data ('ref-data').
    
    Args:
      eval_data: NumPy array of data points from the distribution to be evaluated.
      ref_data: NumPy array of data points from the reference distribution."""

    m1 = np.mean(eval_data, axis=0)
    s1 = np.cov(eval_data, rowvar=False)

    m2 = np.mean(ref_data, axis=0)
    s2 = np.cov(ref_data, rowvar=False)

    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value