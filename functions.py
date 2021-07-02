import numpy as np

def rotate_bar(beta, X, Y, VX, VY):
    X_ = X * np.cos(beta) - Y * np.sin(beta)
    Y_ = X * np.sin(beta) + Y * np.cos(beta)

    VX_ = VX * np.cos(beta) - VY * np.sin(beta)
    VY_ = VX * np.sin(beta) + VY * np.cos(beta)
    return X_, Y_, VX_, VY_

def incline_galaxy(i, X, Y, VX, VY):
    VX_ = VX
    VZ_ = VY * np.sin(i)
    VY_ = VY * np.cos(i)


    X_ = X
    Z_ = Y * np.sin(i)
    Y_ = Y * np.cos(i)
    return X_, Y_, VZ_

def add_solid_body_rotation(X, Y, VX, VY, Omegap=0.4):
    PHI = np.arctan2(Y, X)
    VX -= np.sqrt(X ** 2 + Y ** 2) * Omegap * np.sin(PHI)
    VY += np.sqrt(X ** 2 + Y ** 2) * Omegap * np.cos(PHI)
    return VX, VY
