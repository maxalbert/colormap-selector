import numpy as np
import matplotlib.colors as mcolors
from skimage.color import rgb2lab as rgb2lab_skimage
from skimage.color import lab2rgb as lab2rgb_skimage


class RGBRangeError(Exception):
    pass


def rgb2lab(rgb):
    rgb = np.asarray(rgb).reshape(1, 1, 3)
    lab = rgb2lab_skimage(rgb).reshape(3)
    return lab


def lab2rgb(lab, assert_valid=False, clip=False):
    lab = np.asarray(lab).reshape(1, 1, 3)
    rgb = lab2rgb_skimage(lab).reshape(3)
    if assert_valid and ((rgb < 0.0).any() or (rgb > 1.0).any()):
        raise RGBRangeError()
    if clip:
        rgb = np.clip(rgb, 0., 1.)
    return rgb

def lab2rgba(lab, assert_valid=False, clip=False):
    r, g, b = lab2rgb(lab, assert_valid=assert_valid, clip=clip)
    return np.array([r, g, b, 1.])


def linear_colormap(pt1, pt2, coordspace='RGB'):
    """
    Define a perceptually linear colormap defined through a line in the
    CIELab [1] color space. The line is defined by its endpoints `pt1`,
    `pt2`. The argument `coordspace` can be either `RGB` (the default)
    or `lab` and specifies whether the coordinates of `pt1`, `pt2` are
    given in RGB or Lab coordinates.

    [1] http://dba.med.sc.edu/price/irf/Adobe_tg/models/cielab.html

    """
    if coordspace == 'RGB':
        pt1 = np.array(rgb2lab(pt1))
        pt2 = np.array(rgb2lab(pt2))
    elif coordspace == 'Lab':
        pt1 = np.array(pt1)
        pt2 = np.array(pt2)
    else:
        raise ValueError("Argument 'coordspace' must be either 'RGB' "
                         "or 'Lab'. Got: {}".format(coordspace))

    tvals = np.linspace(0, 1, 256)
    path_vals = np.array([(1-t) * pt1 + t * pt2 for t in tvals])
    cmap_vals = np.array([lab2rgb(pt) for pt in path_vals])
    #print np.where(cmap_vals < 0)
    cmap = mcolors.ListedColormap(cmap_vals)
    return cmap
