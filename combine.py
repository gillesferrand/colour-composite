"""
Generate colour images by combining data layers
method 1: RGB()
method 2: colourize() and blend()
"""

import numpy as np
import matplotlib as mpl

def RGB(layers, weights=[1,1,1], min=None, max=None):
    """Generates a colour image by assigning 3 layers to the R,G,B channels"""
    if len(layers)!=3:
        print "You must provide exactly 3 layers"
        return
    # normalize each layer
    def norm(i,min=min,max=max):
        layer = layers[i].copy()
        if min==None: min = layer.min()
        if max==None: max = layer.max()
        layer = (layer-min)/(max-min) # in [0,1]
        layer *= weights[i]
        return layer
    # stack the layers as R,G,B channels
    channels = []
    for i in range(3): channels.append(norm(i))
    image_RGB = np.stack(channels, axis=2)
    return image_RGB

def colourize(layer, H, S=100, V=100, min=None, max=None):
    """Colorizes a layer by setting Hue angle (in degrees), Saturation (in %), and maximum lightness Value (in %)"""
    channel = {}
    channel['H'] = H/360. * np.ones(layer.shape)
    channel['S'] = S/100. * np.ones(layer.shape)
    if min==None: min = layer.min()
    if max==None: max = layer.max()
    channel['V'] = V/100. * (layer-min)/(max-min)
    image_HSV = np.transpose(np.stack((channel['H'],channel['S'],channel['V'])),axes=(1,2,0))
    image_RGB = mpl.colors.hsv_to_rgb(image_HSV)
    return image_RGB

def blend(layers, mode):
    """Blends any number of coloured `layers` (RGB arrays) using the specified `mode` (a function like the ones listed below)"""
    n = layers[0].shape[0]
    image_RGB = np.zeros((n,n,3))
    for i in range(len(layers)):
        image_RGB = mode(layers[i],image_RGB)
    return image_RGB

# blending modes
# https://helpx.adobe.com/photoshop/using/blending-modes.html
# https://photoblogstop.com/photoshop/photoshop-blend-modes-explained
# A = active layer
# B = background layer
# (assumed to be normalized in [0,1])
def Multiply(A,B): return A*B
def Substract(A,B): return B-A
def Divide(A,B): return B/A
def Add(A,B): return A+B
def LinearDodge(A,B): return A+B
def LinearBurn(A,B): return A+B-1
def ColorDodge(A,B): return B/(1-A)
def ColorBurn(A,B): return 1-(1-B)/A
def Screen(A,B): return 1-(1-A)*(1-B)
