import numpy as np
from cross_section import *


def test_normal_vector_of_plane_has_unit_length():
    plane = Plane([0, 0, 0], [-3, 4, 12])
    np.testing.assert_almost_equal(np.linalg.norm(plane.n), 1.0)


def test_init_cross_section():
    plane = Plane([50, 0, 0], n=[1, 0, 0])
    cs = CrossSection(plane)
    assert isinstance(cs.vertices_3d, np.ndarray)
    assert np.allclose(cs.vertices_3d[:, 0], 50)
    assert isinstance(cs.vertices_2d, np.ndarray)
    assert isinstance(cs.faces, np.ndarray)
    assert isinstance(cs.mapping_3d_to_2d, Mapping3Dto2D)


def test_init_cross_section_L():
    cs = CrossSectionL(L=40)
    cs.L = 10
    assert cs.L == 10
    assert isinstance(cs.vertices_3d, np.ndarray)
    assert np.allclose(cs.vertices_3d[:, 0], 10)
    assert isinstance(cs.vertices_2d, np.ndarray)
    assert isinstance(cs.faces, np.ndarray)
    assert isinstance(cs.mapping_3d_to_2d, Mapping3Dto2D)
