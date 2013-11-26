
"""
TJL Nov 2013
"""

import numpy as np
from scipy import misc, special

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt

from enthought.mayavi import mlab

from odin import xray
from odin.refdata import sph_quad_900
from odin.math2 import arctan3
import mdtraj


def compute_lsqd_coef(Gk):
    """
    Take the output from the optimizer (which has the real & complex components separated)
    and compute the ell-squared coefficient,
    
        ell-squared = sum{ G_{ell m} ^ 2 }
                       m
                       
    """
    
    Gk_mag = np.square( np.abs(Gk) ) # k-indexing now correct
    k_max = len(Gk_mag)
    
    l_max = backward_index(k_max)[0]
    lsqd = np.zeros(l_max+1)
    
    for k in range(k_max):
        l,m = backward_index(k)
        lsqd[l] += Gk_mag[k]
        
    return lsqd


def backward_index(k):
    """
    k --> (ell, m)
    
    Returns ell, m
    """
    ell = int(np.floor(np.sqrt(k)))
    m = int(k - ell*(ell+1))
    return ell, m
    
    
def load_gk_file(filename):
    
    data = np.loadtxt(filename)
    g_k = np.zeros(data.shape[0], dtype=np.complex128)
    g_k = data[:,2] + 1j * data[:,3]
    
    return g_k
    

def evaluate_gk_on_sphere(g_k, xyz):
    """
    Evaluate the density on a sphere defined by a series expansion in spherical
    harmonics (`g_k`) at various points in space (`xyz`). Note that while the
    argument `xyz` is cartesian, the function is evaluated on a sphere (radius
    information is discarded).
    
    Parameters
    ----------
    g_k : np.ndarray, complex
        An array of spherical harmonic coefficients in l/m order.
    
    xyz : np.ndarray, float
        A N x 3 array of positions at which to evaluate the electron density
        parameterized by `g_k`.
    
    Returns
    -------
    density : np.ndarray
        A scalar field, (N,)-shape array, defining the electron density at the
        sample points.
    """
    
    if (not xyz.shape[1] == 3) or (not len(xyz.shape) == 2):
        raise ValueError('`xyz` must be a two dimensional array, and the second'
                         ' dim must be length 3. Got shape: %s' % str(xyz.shape))
    density = np.zeros(xyz.shape[0], dtype=np.complex128)
    
    phi = arctan3(xyz[:,1], xyz[:,0])
    
    for k in range(len(g_k)):
        
        l,m = backward_index(k)
        
        N = np.sqrt( 2. * l * misc.factorial(l-m) / \
                   ( 4. * np.pi * misc.factorial(l+m) ) )
        
        Plm = special.lpmv(m, l, xyz[:,2])
        Ylm = N * np.exp( 1j * m * phi ) * Plm
        
        density += g_k[k] * Ylm
    
    return density


def plot_gk_on_sphere(g_k):

    r = 0.3
    phi, theta = np.mgrid[0:np.pi:101j, 0:2*np.pi:101j]

    x = r*np.sin(phi)*np.cos(theta)
    y = r*np.sin(phi)*np.sin(theta)
    z = r*np.cos(phi)
    
    xyz = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T

    density = evaluate_gk_on_sphere(g_k, xyz)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    mlab.mesh(x, y, z, 
              scalars=np.imag(density).reshape(x.shape), colormap='jet')
    mlab.show()
    
    return
    
def plot_spectrum_slice(g_k):
    
    Gk_mag = np.square(np.abs(g_k))
    
    plt.figure()
    
    plt.subplot(121)
    plt.plot(Gk_mag)
    
    plt.subplot(122)
    plt.plot( compute_lsqd_coef(g_k) )
    
    plt.show()
    
    return
    

# x, y, z = np.ogrid[-10:10:20j, -10:10:20j, -10:10:20j]
# s = np.sin(x*y*z)/(x*y*z)
# 
# src = mlab.pipeline.scalar_field(s)
# mlab.pipeline.iso_surface(src, contours=[s.min()+0.1*s.ptp(), ], opacity=0.3)
# mlab.pipeline.iso_surface(src, contours=[s.max()-0.1*s.ptp(), ],)
# 
# mlab.show()

if __name__ == '__main__':
    
    g_k = load_gk_file('Gk_v.dat')
    plot_gk_on_sphere(g_k)
    # plot_spectrum_slice(g_k)
    
   
