# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 12:32:37 2019

@author: Rob

helper file for MNIST data
"""
import matplotlib.pyplot as plt
import numpy as np

def MNIST_Plot(pixels, title = ''):

    pixels = np.array(pixels)
    
    # Reshape the array into 28 x 28 array (2-dimensional array)
    pixels = pixels.reshape((28, 28))
    
    # Plot
    plt.title(title)
    plt.imshow(pixels, cmap='gray')
    plt.show()
