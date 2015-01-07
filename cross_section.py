import numpy as np
import itertools
from scipy.spatial import Delaunay
from color_transformations import rgb2lab, lab2rgb
from mapping_3d_to_2d import Mapping3Dto2D


class NoIntersectionError(Exception):
    # Used when computing intersections of line segments with a plane.
    pass


def compute_intersection_of_image_curve_with_plane(P1, P2, plane, fun=rgb2lab, TOL=1e-6):
    """
    Given two points `P1`, `P2` in RGB space and a transformation
    function `fun` (e.g. from RGB space to CIELab space), find the
    intersection of the curve which is the image of the line segment
    `P1`, `P2` under `fun` with a given plane in the image space.

    """
    P1 = np.asarray(P1, dtype=float)
    P2 = np.asarray(P2, dtype=float)

    Q1 = fun(P1)
    Q2 = fun(P2)

    # If the image points are close enough, we are done
    if np.linalg.norm(Q1 - Q2) < TOL:
        return 0.5 * (Q1 + Q2)

    # If the image points lie on the same side of the plane, we assume
    # that there is no intersection (this will not be true for some
    # positions of cross section planes through CIELab space, but is a
    # good assumption for most cases of interest to us).
    if plane.same_side(Q1, Q2):
        raise NoIntersectionError()

    P3 = 0.5 * (P1 + P2)
    Q3 = fun(P3)

    if plane.same_side(Q2, Q3):
        return compute_intersection_of_image_curve_with_plane(P1, P3, plane, fun=fun, TOL=TOL)
    elif plane.same_side(Q1, Q3):
        return compute_intersection_of_image_curve_with_plane(P3, P2, plane, fun=fun, TOL=TOL)
    else:
        raise RuntimeError("This should not happen!")


class Plane(object):
    def __init__(self, P, n):
        """
        Initialise plane from an incident point `P` and a normal vector `n`.

        """
        self.pt = np.asarray(P, dtype=float)
        self.n = np.asarray(n, dtype=float)
        self.n /= np.linalg.norm(n)

    def same_side(self, P1, P2):
        """
        Return `True` if the points `P1` and `P2` lie on the same side
        of the plane, otherwise return `False`. If one of the points
        lies exactly in the plane then `False` is returned.

        """
        a = np.dot(self.n, self.pt - P1)
        b = np.dot(self.n, self.pt - P2)
        return (a * b > 0)


class CrossSection(object):
    """
    Cross section through the set of all "RGB-representable" colors in
    a given color space.

    """
    color_transformations = {
        'CIELab': rgb2lab
        }

    def __init__(self, plane, color_space='CIELab'):
        self.vertices_3d = None
        self.vertices_2d = None
        self.vertex_colors = None
        self.faces = None
        self.color_space = color_space
        self.set_plane(plane)

    def set_plane(self, plane):
        self.plane = plane
        self.mapping_3d_to_2d = Mapping3Dto2D(self.plane)
        self.compute_triangulation()

    @property
    def color_space(self):
        return self._color_space

    @color_space.setter
    def color_space(self, value):
        if value in self.color_transformations.keys():
            self._color_space = value
        else:
            raise ValueError(
                "Unsupported color space: '{}'. Must be one of: {}"
                "".format(value, self.color_transformations.keys()))

    def compute_triangulation(self, N=5):
        """
        Compute a triangulation of the cross section.

        This works by subdividing the RGB cube into line segments
        parallel to each coordinate axis, transforming these line
        segments using the functon `fun` and computing the
        intersections of the resulting curves with `plane`. The
        argument `N` specifies the number of subdivisions of each edge
        of the RGB cube.

        """
        fun = self.color_transformations[self.color_space]

        # Compute the intersections of the images in CIELab space of a
        # bunch of line segments parallel to the coordinate axes with the
        # cross section plane.
        pts_intersection = []

        def add_intersection_point(P1, P2):
            try:
                Q = compute_intersection_of_image_curve_with_plane(P1, P2, self.plane, fun=fun)
                pts_intersection.append(Q)
            except NoIntersectionError:
                pass

        vals = np.linspace(0., 1., N)
        for a, b in itertools.product(vals, vals):
            add_intersection_point([a, b, 0.], [a, b, 1.])
            add_intersection_point([a, 0., b], [a, 1., b])
            add_intersection_point([0., a, b], [1., a, b])

        self.vertices_3d = np.array(pts_intersection)
        self.vertices_2d = self.mapping_3d_to_2d.apply(pts_intersection)

        # # The points are coplanar but the plane they lie in has general
        # # position in space. Here we rotate the points so that one
        # # coordinate is constant and drop this coordinate. This allows us
        # # to determine a 2D Delaunay triangulation of the point cloud
        # # below.
        # pts_2d, _, _, _ = find_planar_coordinates(pts_intersection)

        # Compute Delaunay triangulation of the now 2D point set
        tri = Delaunay(self.vertices_2d)
        self.faces = tri.simplices

        # Compute vertex colors (RGBA)
        self.vertex_colors = np.empty((len(self.vertices_3d), 4))
        self.vertex_colors[:, 0:3] = np.array([lab2rgb(pt) for pt in self.vertices_3d])
        self.vertex_colors[:, 3] = 0.7  # alpha value != 1 for slight transparency


class CrossSectionL(CrossSection):
    def __init__(self, L):
        super(CrossSectionL, self).__init__(Plane([L, 0, 0], n=[1, 0, 0]))
        self._L = L

    @property
    def L(self):
        return self._L

    @L.setter
    def L(self, value):
        self._L = value
        self.set_plane(Plane([value, 0, 0], n=[1, 0, 0]))
