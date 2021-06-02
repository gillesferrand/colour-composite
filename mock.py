"""
Generate mock clumpy 2D density fields
method 1: Gaussian clumps
method 2: Gaussian fields
"""

import numpy as np

#------------------------------
# by splatting Gaussian clumps
#------------------------------

def Gaussian_clump(sigma, halfsize):
    """Draws a Gaussian clump of standard deviation `sigma` on a 2D box of `halfsize`"""
    size = 2*halfsize+1
    clump = np.zeros((size,size))
    for ix in range(size):
        for iy in range(size):
            d2 = (ix-halfsize)**2 + (iy-halfsize)**2
            clump[ix,iy] = np.exp(-d2/float(sigma)**2)
    return clump

def add_block(b1, b2, pos_v, pos_h):
    """Adds array block `b2` to array `b1`"""
    v_range1 = slice(max(0, pos_v), max(min(pos_v + b2.shape[0], b1.shape[0]), 0))
    h_range1 = slice(max(0, pos_h), max(min(pos_h + b2.shape[1], b1.shape[1]), 0))
    v_range2 = slice(max(0, -pos_v), min(-pos_v + b1.shape[0], b2.shape[0]))
    h_range2 = slice(max(0, -pos_h), min(-pos_h + b1.shape[1], b2.shape[1]))
    b1[v_range1, h_range1] += b2[v_range2, h_range2]

def Gaussian_clumps(n_clumps, nx, ny=0, amp=[1], sigma=[], x=[], y=[], fact=3, verbose=False):
    """
    Generates a 2D array of size `nx` by `ny` with `n_clumps` clumps of amplitude `amp` and std dev `sigma` at positions `x`,`y`
    Lengths `x`, `y`, `sigma` are expressed as a fraction of the box lengths nx, ny, min(nx,ny)
    If arrays `amp`, `sigma`, `x`, `y` are of length 1 they are repeated to match `n_clumps`
    If they are left empty they are chosen randomly in [0,1]
    """
    if ny==0: ny=nx
    s = 2*int(sigma[0]*min(nx,ny)) if len(sigma)==1 else 0
    if len(amp  )==1: amp   = amp  *n_clumps
    if len(sigma)==1: sigma = sigma*n_clumps
    if len(x    )==1: x     = x    *n_clumps
    if len(y    )==1: y     = y    *n_clumps
    if len(amp  )==0: amp   = np.random.rand(n_clumps)
    if len(sigma)==0: sigma = np.random.rand(n_clumps)
    if len(x    )==0: x     = np.random.rand(n_clumps)
    if len(y    )==0: y     = np.random.rand(n_clumps)
    arr = np.zeros((nx,ny))
    for ic in range(n_clumps):
        isigma = int(sigma[ic]*min(nx,ny))
        if isigma>0:
            clump = amp[ic] * Gaussian_clump(isigma, fact*isigma)
            pos_v = int(x[ic]*nx) - fact*isigma
            pos_h = int(y[ic]*ny) - fact*isigma
            add_block(arr, clump, pos_v, pos_h)
    if verbose and s>0: print "typical size = %3i -> f_max = %.1f"%(s,min(nx,ny)/float(2*s))
    return arr

#------------------------------
# by inverse Fourier transform
#------------------------------
# from http://andrewwalker.github.io/statefultransitions/post/gaussian-fields/

def fftIndgen(n):
    a = range(0, n/2+1)
    b = range(1, n/2)
    b.reverse()
    b = [-i for i in b]
    return a + b

def Gaussian_field(Pk = lambda k : k**-3.0, nx=128, ny=128):
    """Generates a 2D array of size `nx` by `ny` with scale distribution given by function `Pk`"""
    def Pk2(kx, ky):
        kr = np.sqrt(kx**2 + ky**2)
        if kr == 0: return 0.0
        return Pk(kr)
    amplitude = np.zeros((nx,ny),dtype='complex128')
    for i, kx in enumerate(fftIndgen(nx)):
        for j, ky in enumerate(fftIndgen(ny)):
            amplitude[i, j] = np.sqrt(Pk2(kx, ky))
    noise = np.fft.fft2(np.random.normal(size = (nx, ny)))
    field = np.fft.ifft2(noise*amplitude)
    return field.real
