"""
Generate colour images by combining data layers
method 1: RGB()
method 2: colourize() and blend()
"""

from __future__ import print_function
import numpy as np
import matplotlib as mpl

#-------------------------------------------------------------------------------

def norm(layer, min=None, max=None):
    """ Normalizes a layer from [min,max] to [0,1]
        (if min/max is not set, it is taken from the data)
    """
    if min==None: min = layer.min()
    if max==None: max = layer.max()
    if max-min==0:
        print("min must be different from max")
        return
    layer_norm = (layer.copy()-min)/(max-min)
    return layer_norm

def RGB(layers, weights=[1], min=[None], max=[None]):
    """ Generates a colour image by assigning 3 layers to the R,G,B channels.
        Each layer is first normalized from [min, max] to [0,1], and optionally weighted.
    """
    if len(layers)!=3:
        print("You must provide exactly 3 layers")
        return
    if len(weights)==1: weights = weights *3
    if len(min)    ==1: min     = min     *3
    if len(max)    ==1: max     = max     *3
    channels = []
    for i in range(3):
        channel = weights[i] * norm(layers[i], min[i], max[i])
        channels.append(channel)
    image_RGB = np.stack(channels, axis=2)
    return image_RGB

#-------------------------------------------------------------------------------

def colourize_RGB(layer, R, G, B, min=None, max=None):
    """ Colourizes a layer by setting the weight of the R,G,B channels
    """
    image_RGB = RGB([layer, layer, layer], weights=[R, G, B], min=[min], max=[max])
    return image_RGB

def colourize_HSV(layer, H, S, V, min=None, max=None):
    """ Colourizes a layer by setting:
        H = hue angle (in degrees)
        S = saturation (in %)
        V = maximum lightness Value (in %)
    """
    channel = {}
    channel['H'] = H/360. * np.ones(layer.shape)
    channel['S'] = S/100. * np.ones(layer.shape)
    if min==None: min = layer.min()
    if max==None: max = layer.max()
    channel['V'] = V/100. * (layer-min)/(max-min)
    image_HSV = np.transpose(np.stack((channel['H'],channel['S'],channel['V'])),axes=(1,2,0))
    image_RGB = mpl.colors.hsv_to_rgb(image_HSV)
    return image_RGB

def colourize_list(colourizer, layers, X, Y, Z, min=[None], max=[None]):
    """ Colourizes a list of layers using the given `colourizer` function
        (If of length one, other parameters will be padded to match `layers`.)
    """
    if len(X  )==1: X   = X   *len(layers)
    if len(Y  )==1: Y   = Y   *len(layers)
    if len(Z  )==1: Z   = Z   *len(layers)
    if len(min)==1: min = min *len(layers)
    if len(max)==1: max = max *len(layers)
    colourized_layers = []
    for j in range(len(layers)):
        colourized_layer = colourizer(layers[j], X[j], Y[j], Z[j], min[j], max[j])
        colourized_layers.append(colourized_layer)
    return colourized_layers

def colourize_RGB_list(layers, R, G, B, min=[None], max=[None]):
    """ Colourizes a list of layers in RGB
    """
    return colourize_list(colourize_RGB, layers, R, G, B, min=[None], max=[None])

def colourize_HSV_list(layers, H, S, V, min=[None], max=[None]):
    """ Colourizes a list of layers in HSV
    """
    return colourize_list(colourize_HSV, layers, H, S, V, min=[None], max=[None])

#-------------------------------------------------------------------------------

def blend(layers, mode, norm=True):
    """ Blends any number of coloured `layers` using the specified `mode`.
       `layers` is a list of (n,m,3) arrays
       `mode` is a function of 2 layers, like the ones listed below
    """
    n = layers[0].shape[0]
    m = layers[0].shape[1]
    image_RGB = np.zeros((n,m,3))
    for i in range(3):
        for j in range(len(layers)):
            image_RGB[:,:,i] = mode(layers[j][:,:,i], image_RGB[:,:,i])
        if norm: image_RGB[:,:,i] /= image_RGB[:,:,i].max()
    return image_RGB

# blending modes
# https://helpx.adobe.com/photoshop/using/blending-modes.html
# https://photoblogstop.com/photoshop/photoshop-blend-modes-explained
# A = active layer
# B = background layer
# (both layers are assumed to be normalized in [0,1])
# (operations are meant to be done element-wise, which is the case for numpy arrays)
def Multiply(A,B): return A*B
def Substract(A,B): return B-A
def Divide(A,B): return B/A
def Add(A,B): return A+B
def LinearDodge(A,B): return A+B
def LinearBurn(A,B): return A+B-1
def ColorDodge(A,B): return B/(1-A)
def ColorBurn(A,B): return 1-(1-B)/A
def Screen(A,B): return 1-(1-A)*(1-B)
