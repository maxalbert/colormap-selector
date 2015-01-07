import numpy as np

whitepoint_D65 = np.array([0.9642, 1, 0.8249])

A_xyz2rgb = np.array(
   [[3.240479, -1.537150, -0.498535],
    [-0.969256, 1.875992,  0.041556 ],
    [0.055648, -0.204043,  1.057311 ]])

A_rgb2xyz = np.linalg.inv(A_xyz2rgb)


class RGBRangeError(Exception):
    pass


def f(t):
    if t > (6./29)**3:
        return t**(1./3)
    else:
        return 1./3 * (29./6)**2 * t + 4./29


def f_inv(t):
    if t > 6./29:
        return t**3
    else:
        return 3 * (6./29)**2 * (t - 4./29)


def xyz2lab(xyz, whitepoint=whitepoint_D65):
    """
    Convert from CIELAB to XYZ color coordinates.

    *Arguments*

    xyz:  3-tuple (or other list-like)

    *Returns*

    3-tuple (L, a, b).

    """
    X, Y, Z = xyz
    Xw, Yw, Zw = whitepoint

    L = 116. * f(Y/Yw) - 16
    a = 500. * (f(X/Xw) - f(Y/Yw))
    b = 200. * (f(Y/Yw) - f(Z/Zw))

    return np.array([L, a, b], dtype=float)


def lab2xyz(lab, whitepoint=whitepoint_D65):
    L, a, b = lab
    Xw, Yw, Zw = whitepoint

    Y = Yw * f_inv(1./116 * (L + 16))
    X = Xw * f_inv(1./116 * (L + 16) + 0.002 * a)
    Z = Zw * f_inv(1./116 * (L + 16) - 0.005 * b)

    return X, Y, Z


def rgb2xyz(rgb):
    rgb = np.asarray(rgb)
    return np.dot(A_rgb2xyz, rgb)


def xyz2rgb(xyz, assert_valid=False, clip=False):
    xyz = np.asarray(xyz)
    rgb = np.dot(A_xyz2rgb, xyz)
    r, g, b = rgb
    if assert_valid and ((r < 0.0 or r > 1.0) or
                         (g < 0.0 or g > 1.0) or
                         (b < 0.0 or b > 1.0)):
        raise RGBRangeError()
    if clip:
        rgb = np.clip(rgb, 0., 1.)
    return rgb


def rgb2lab(rgb, whitepoint=whitepoint_D65):
    return xyz2lab(rgb2xyz(rgb), whitepoint=whitepoint)


def lab2rgb(lab, whitepoint=whitepoint_D65, assert_valid=False, clip=False):
    return xyz2rgb(lab2xyz(lab, whitepoint=whitepoint), assert_valid=assert_valid, clip=clip)


def rgb2rgba(rgb):
    r, g, b = rgb
    return np.array([r, g, b, 1.])

def lab2rgba(lab, whitepoint=whitepoint_D65, assert_valid=False, clip=False):
    return rgb2rgba(lab2rgb(lab, whitepoint=whitepoint, assert_valid=assert_valid, clip=clip))
