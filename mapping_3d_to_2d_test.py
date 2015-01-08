import numpy as np
from cross_section import Plane
from mapping_3d_to_2d import *


def test_initialise_mapping_3d_to_2d_simple():
    """
    Check that for a plane orthogonal to the x-axis the transformation
    simply drops the constant x-coordinate.
    """
    plane1 = Plane([50, 0, 0], n=[1, 0, 0])
    f1 = Mapping3Dto2D(plane1)

    # Check the 3d -> 2d transformation
    assert np.allclose(f1.apply([50, -1, 4]), [-1, 4])
    assert np.allclose(f1.apply([50, 3, 7]), [3, 7])

    # Check the 2d -> 3d transformation
    assert np.allclose(f1.apply_inv([-1, 4]), [50, -1, 4])
    assert np.allclose(f1.apply_inv([3, 7]), [50, 3, 7])
    assert f1.apply_inv([-1, 4]).ndim == 1
    assert f1.apply_inv([[-1, 4]]).ndim == 2
    assert np.allclose(f1.apply_inv([[-1, 4], [3, 7]]), [[50, -1, 4], [50, 3, 7]])

    # Regression test: check that applying the transformation does not
    # change the shape/dimension of the input array.
    pt1 = np.array([2., 6., 4.])
    pt2 = np.array([[2., 6., 4.]])
    _ = f1.apply(pt1)
    _ = f1.apply(pt2)
    assert pt1.shape == (3,)
    assert pt2.shape == (1, 3)

    plane2 = Plane([0, 30, 0], n=[0, 1, 0])
    f2 = Mapping3Dto2D(plane2)

    # Check the 3d -> 2d transformation
    assert np.allclose(f2.apply([-1, 30, 4]), [-1, 4])
    assert np.allclose(f2.apply([3, 30, 7]), [3, 7])

    # Check the 2d -> 3d transformation
    assert np.allclose(f2.apply_inv([-1, 4]), [-1, 30, 4])
    assert np.allclose(f2.apply_inv([3, 7]), [3, 30, 7])

    # Regression test: check that applying the inverse transformation
    # does not change the shape/dimension of the input array.
    pt1 = np.array([2., 6.])
    pt2 = np.array([[2., 6.]])
    _ = f1.apply_inv(pt1)
    _ = f1.apply_inv(pt2)
    assert pt1.shape == (2,)
    assert pt2.shape == (1, 2)
