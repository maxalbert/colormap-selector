import numpy as np
import matplotlib.colors as mcolors
from scipy.spatial import Delaunay
import sys
import itertools
import matplotlib.pyplot as plt
from PyQt4 import QtGui, QtCore
from vispy import app, scene, gloo
from vispy.scene.visuals import Line, Mesh, Markers


whitepoint_D65 = np.array([0.9642, 1, 0.8249])

A_xyz2rgb = np.array(
   [[3.240479, -1.537150, -0.498535],
    [-0.969256, 1.875992,  0.041556 ],
    [0.055648, -0.204043,  1.057311 ]])

A_rgb2xyz = np.linalg.inv(A_xyz2rgb)


class RGBRangeError(Exception):
    pass


class NoIntersectionError(Exception):
    # Used when computing intersections of line segments with a plane.
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


def transform_mesh(mesh, f):
    """
    Return a new mesh where the vertex coordinates have been
    transformed with the function `f`.

    """
    editor = df.MeshEditor()
    mesh_new = df.Mesh()
    editor.open(mesh_new, mesh.topology().dim(), mesh.geometry().dim())  # top. and geom. dimension are both 3
    editor.init_vertices(mesh.num_vertices())
    editor.init_cells(mesh.num_cells())
    for i, pt in enumerate(mesh.coordinates()):
        editor.add_vertex(i, np.array(f(pt)))
    for i, c in enumerate(mesh.cells()):
        editor.add_cell(i, np.array(c, dtype=np.uintp))
    editor.close()
    return mesh_new


def uniqify(seq):
    """
    Return a copy of the sequence `seq` with duplicate elements removed.
    This is the function 'f7' from [1].

    [1] http://www.peterbe.com/plog/uniqifiers-benchmark
    """
    seen = set()
    seen_add = seen.add
    return [ x for x in seq if not (x in seen or seen_add(x))]


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation
    about the given axis by theta radians.

    """
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(0.5*theta)
    b, c, d = -axis*np.sin(0.5*theta)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])


def find_planar_coordinates(pts):
    """
    Given a set of points `pts` contained in a plane that lies in
    general position in 3D space, rotate this set into a plane that
    has a fixed x, y or z coordinate and drop the constant axis.

    *Arguments*

    pts:

        Sequence of 3d points (must be coplanar).


    *Returns*

    A tuple (coords, idx, c0, R) where `coords` is a set of 2D coordinates,
    `idx` is the index of the constant axis (0, 1, 2 for x, y, z axis,
    respectively), `c0` is the value of the constant coordinate of the
    rotated points and `R` is the rotation matrix used to map the original
    points to the new ones.

    """
    pts = np.asarray(pts)
    assert pts.ndim == 2
    assert pts.shape[1] == 3

    # Find a normal vector 'v' of the plane in which the points lie
    P1, P2, P3 = pts[:3]
    v = np.cross(P2-P1, P3-P1)

    # Find the coordinate axis `w` which has the largest possible angle
    # with respect to v. This is the axis whose dot product with v is
    # smallest, i.e. the axis along which v has its minimum coordinate.
    idx = np.argmin(np.absolute(v))
    w = np.array([0., 0., 0.])
    w[idx] = 1.0

    # We rotate the points into a plane with constant idx-coordinates,
    # i.e. a plane orthogonal to the vector w. The rotation axis is
    # given by the cross product of v and w and the angle by their dot
    # product (after normalising).
    rot_axis = np.cross(v, w)
    theta = np.arccos(np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w)))  # rotation angle
    R = rotation_matrix(rot_axis, theta)
    pts_rotated = np.array([np.dot(R, pt) for pt in pts])

    # Verify that the idx-coordinate is constant
    const_coords = pts_rotated[:, idx]
    c0 = const_coords[0]
    if not np.allclose(const_coords, c0):
        raise RuntimeError("The given points are not coplanar!")

    # Drop the idx-column from the point coordinates
    indices = [i for i in xrange(3) if i != idx]
    pts_2d = pts_rotated[:, indices]

    return pts_2d, idx, c0, R


def create_mesh(vertices, simplices):
    vertices = np.asarray(vertices, dtype=float)
    simplices = np.asarray(simplices, dtype=np.uintp)
    assert vertices.ndim == 2
    assert simplices.ndim == 2

    top_dim = simplices.shape[1] - 1  # topological dimension of the mesh
    _, geom_dim = vertices.shape  # geometrical dimension of the mesh

    # Create a mesh from the points in 3D space by using the
    # connectivity information from the Delaunay triangulation.
    editor = df.MeshEditor()
    mesh = df.Mesh()
    editor.open(mesh, top_dim, geom_dim)
    editor.init_vertices(len(vertices))
    editor.init_cells(len(simplices))
    for i, pt in enumerate(vertices):
        editor.add_vertex(i, pt)
    for i, c in enumerate(simplices):
        editor.add_cell(i, c)
    editor.close()

    return mesh


def create_quadrilateral_mesh(A, B, C, D):
    """
    Create a mesh consisting of two triangles connecting the four given points.
    The two triangles are [A, B, C] and [A, C, D], thus the points must be
    given in the correct order to create a full quadrilateral (rather than two
    overlapping triangles).
    """
    return create_mesh(vertices=[A, B, C, D], simplices=[[0, 1, 2], [0, 2, 3]])


def cross_section_triangulation_from_meshes(mesh_3d, mesh_2d):
    """
    Given a tetrahedral mesh `mesh_3d` and a triangular mesh `mesh_2d`
    (which must be embedded in 3-space), compute a triangulation of
    the cross section between the two meshes.

    Known limitations: Currently the triangular mesh is required to be planar.

    """
    cells = [c for c in df.cells(mesh_3d)]

    # Compute the intersections of each triangle in the 2d mesh with
    # all cells in the 3d mesh.
    pts_intersection = []
    triangles = [c for c in df.cells(mesh_2d)]
    for triangle in triangles:
        # Determine a unique list of pointer of intersection between
        # the triangle and the mesh edges.
        pts_cur = [c.triangulate_intersection(triangle).reshape(-1, 3) for c in cells]
        pts_cur = [pt for pt in pts_cur if pt.size != 0]
        if pts_cur != []:
            pts_intersection.append(np.concatenate(pts_cur))
    pts_intersection = np.concatenate(pts_intersection)
    pts_intersection = np.array(uniqify([tuple(pt) for pt in pts_intersection]))

    # The points are coplanar but the plane they lie in has general
    # position in space. Here we rotate the points so that one
    # coordinate is constant and drop this coordinate. This allows us
    # tod determine a 2D Delaunay trian- gulation of the point cloud
    # below.
    pts_2d, _, _, _ = find_planar_coordinates(pts_intersection)

    # Compute Delaunay triangulation of the now 2D point set
    tri = Delaunay(pts_2d)

    return pts_intersection, tri.simplices


def cross_section_triangulation(plane, N=5, fun=rgb2lab):
    """
    Compute a triangulation of the cross section defined by `plane`.

    This works by subdividing the RGB cube into line segments parallel
    to each coordinate axis, transforming these line segments using
    the functon `fun` and computing the intersections of the resulting
    curves with `plane`. The argument `N` specifies the number of
    subdivisions of each edge of the RGB cube.

    """
    # Compute the intersections of the images in CIELab space of a
    # bunch of line segments parallel to the coordinate axes with the
    # cross section plane.
    pts_intersection = []

    vals = np.linspace(0., 1., N)

    def add_intersection_point(P1, P2):
        try:
            Q = compute_intersection_of_image_curve_with_plane(P1, P2, plane, fun=fun)
            pts_intersection.append(Q)
        except NoIntersectionError:
            pass

    for a, b in itertools.product(vals, vals):
        add_intersection_point([a, b, 0.], [a, b, 1.])
        add_intersection_point([a, 0., b], [a, 1., b])
        add_intersection_point([0., a, b], [1., a, b])

    pts_intersection = np.array(pts_intersection)

    # The points are coplanar but the plane they lie in has general
    # position in space. Here we rotate the points so that one
    # coordinate is constant and drop this coordinate. This allows us
    # to determine a 2D Delaunay triangulation of the point cloud
    # below.
    pts_2d, _, _, _ = find_planar_coordinates(pts_intersection)

    # Compute Delaunay triangulation of the now 2D point set
    tri = Delaunay(pts_2d)

    return pts_intersection, tri.simplices


class Plane(object):
    def __init__(self, P, n):
        """
        Initialise plane from a point `P` and a normal vector `n`.

        """
        self.P = np.asarray(P, dtype=float)
        self.n = np.asarray(n, dtype=float)

    def same_side(self, P1, P2):
        """
        Return `True` if the points `P1` and `P2` lie on the same side
        of the plane, otherwise return `False`. If one of the points
        lies exactly in the plane then `False` is returned.

        """
        a = np.dot(self.n, self.P - P1)
        b = np.dot(self.n, self.P - P2)
        return (a * b > 0)


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


class CrossSection(object):
    def __init__(self, plane):
        self.vertices, self.simplices = cross_section_triangulation(plane, N=5)
        #self.vertices = np.asarray(vertices, dtype=float)
        #self.simplices = np.asarray(simplices, dtype=np.uintp)
        self.compute_2d_representation()

    def _equalize_axis_ranges(self, ax, coords):
        is_3d = (coords.shape[1] == 3)
        x = coords[:, 0]
        y = coords[:, 1]

        # Set min/max values along each coordinate axis so that
        # each axis covers the same range (otherwise the plot
        # can look quite distorted).
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        xdiff = xmax - xmin
        ydiff = ymax - ymin
        if is_3d:
            z = coords[:, 2]
            zmin, zmax = z.min(), z.max()
            zdiff = zmax - zmin
        else:
            zdiff = 0.0
        delta = 0.55 * np.max([xdiff, ydiff, zdiff])
        xmin, xmax = 0.5*(xmin+xmax) - delta, 0.5*(xmin+xmax) + delta
        ymin, ymax = 0.5*(ymin+ymax) - delta, 0.5*(ymin+ymax) + delta
        if is_3d:
            zmin, zmax = 0.5*(zmin+zmax) - delta, 0.5*(zmin+zmax) + delta

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        if is_3d:
            ax.set_zlim(zmin, zmax)

    def map_2d_to_3d(self, pt):
        """
        Store the map which transforms 2d coordinates back into the
        original 3d coordinates. This is needed to find the correcte
        CIELab coordinates corresponding to a mouse click event.
        """

        pt3 = np.empty(3)
        pt3[self._idx1] = pt[0]
        pt3[self._idx2] = pt[1]
        pt3[self._idx] = self._z0
        return np.dot(self._Rinv, pt3)

    def compute_2d_representation(self):
        self._vertices_2d, self._idx, self._z0, self._R = \
            find_planar_coordinates(self.vertices)

        print("[DDD] Computing 2D representation. self._idx: {}, self._z0: {}".format(self._idx, self._z0))
        self._idx1, self._idx2 = [i for i in range(3) if i != self._idx]
        self._Rinv = np.linalg.inv(self._R)

    @property
    def vertices_2d(self):
        if self._vertices_2d is None:
            self.compute_2d_representation()
        return self._vertices_2d

    def get_2d_representation(self):
        pts_2d, idx, z0, _ = find_planar_coordinates(self.vertices)
        pts_2d_embedded = np.zeros((len(pts_2d), 3))
        pts_2d_embedded[:, 0:2] = pts_2d
        return pts_2d_embedded, self.simplices

    def plot3d(self, ax=None, show_grid=False, elev=None, azim=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
        else:
            if not isinstance(ax, Axes3D):
                raise TypeError("Argument 'ax' must be of type Axes3D")

        coords = self.vertices
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]

        verts = np.array([np.array([[x[T[0]], y[T[0]], z[T[0]]],
                                    [x[T[1]], y[T[1]], z[T[1]]],
                                    [x[T[2]], y[T[2]], z[T[2]]]]) for T in self.simplices])
        midpoints = np.average(verts, axis=1)
        facecolors = [np.clip(lab2rgb(pt), 0., 1.) for pt in midpoints]

        plot_triangulation(self.vertices, self.simplices, facecolors, ax=ax, show_grid=show_grid)

        self._equalize_axis_ranges(ax, self.vertices)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_aspect('equal')
        ax.view_init(elev=elev, azim=azim)

        return ax.figure

    #def find_planar_coordinates(self, pts):
    #    return find_planar_coordinates(pts)

    def plot2d(self, ax=None, show_grid=False):
        if ax is None:
            fig = plt.figure()
            ax = fig.gca()

        pts_2d, idx, z0, _ = find_planar_coordinates(self.vertices)

        x = pts_2d[:, 0]
        y = pts_2d[:, 1]

        verts = np.array([np.array([[x[T[0]], y[T[0]]],
                                    [x[T[1]], y[T[1]]],
                                    [x[T[2]], y[T[2]]]]) for T in self.simplices])
        midpoints = np.average(verts, axis=1)

        # TODO: Check that 'transform_pt' puts the constant coordinate in the correct position!
        idx1, idx2 = [i for i in [0, 1, 2] if i != idx]
        def transform_pt(z0, a, b):
            pt = np.array([0, 0, 0])
            pt[idx] = z0
            pt[idx1] = a
            pt[idx2] = b
            return pt
        #facecolors = [lab2rgb([z0, a, b]) for (a, b) in midpoints]
        facecolors = [lab2rgb(transform_pt(z0, a, b)) for (a, b) in midpoints]
        facecolors = np.clip(facecolors, 0.0, 1.0)

        coll = PolyCollection(verts, facecolors=facecolors, edgecolors=facecolors if not show_grid else 'black')
        ax.add_collection(coll)

        self._equalize_axis_ranges(ax, pts_2d)
        ax.set_aspect('equal')

        def onclick(event):
            print("[DDD] Plot clicked!")
            a, b = event.xdata, event.ydata
            ax.plot(a, b, 'o', markersize=30)
            ax.figure.canvas.draw()

        cid = ax.figure.canvas.mpl_connect('button_press_event', onclick)

        return ax.figure


class CrossSectionL(CrossSection):
    def __init__(self, L):
        self.set_L(L)

    def set_L(self, L):
        self.L = L
        self.plane = Plane([L, 0, 0], n=[1, 0, 0])
        self.vertices, self.simplices = cross_section_triangulation(self.plane, N=5)
        self.compute_2d_representation()

    def compute_2d_representation(self):
        self._vertices_2d = self.vertices[:, 1:3]
        self._idx = 0
        self._idx1 = 1
        self._idx2 = 2
        self._z0 = self.L
        self._R = np.eye(3)
        self._Rinv = np.eye(3)

    #def find_planar_coordinates(pts):
    #    pts_2d = pts[:, 1:3]
    #    return pts_2d, 0., self.L, None

    def plot2d(self, ax=None, show_grid=False):
        if ax is None:
            fig = plt.figure()
            ax = fig.gca()

        #pts_2d, idx, z0, _ = find_planar_coordinates(self.vertices)
        pts_2d = self.vertices[:, [1, 2]]

        x = pts_2d[:, 0]
        y = pts_2d[:, 1]

        verts = np.array([np.array([[x[T[0]], y[T[0]]],
                                    [x[T[1]], y[T[1]]],
                                    [x[T[2]], y[T[2]]]]) for T in self.simplices])
        midpoints = np.average(verts, axis=1)

        # FIXME: Check 'idx' to find the correct constant coordinate (dosn't have to be L)
        facecolors = [lab2rgb([self.L, a, b]) for (a, b) in midpoints]
        facecolors = np.clip(facecolors, 0.0, 1.0)

        coll = PolyCollection(verts, facecolors=facecolors, edgecolors=facecolors if not show_grid else 'black')
        ax.add_collection(coll)

        self._equalize_axis_ranges(ax, pts_2d)
        ax.set_aspect('equal')

        #def onclick(event):
        #    print("[DDD] Plot clicked!")
        #    a, b = event.xdata, event.ydata
        #    ax.plot(a, b, 'o', markersize=30)
        #    ax.figure.canvas.draw()

        #cid = ax.figure.canvas.mpl_connect('button_press_event', onclick)

        return ax.figure


class SliderWithLabel(QtGui.QWidget):
    def __init__(self, label, *args, **kwargs):
        QtGui.QWidget.__init__(self, *args, **kwargs)
        self.slider = QtGui.QSlider()
        self.slider.setOrientation(QtCore.Qt.Horizontal)
        self.slider.setRange(1, 99)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(10)
        self.label = QtGui.QLabel(self)
        self.label.setText(label)
        # QVBoxLayout the label above; could use QHBoxLayout for
        # side-by-side
        layout = QtGui.QHBoxLayout()
        layout.setMargin(0)
        layout.setSpacing(2)
        layout.addWidget(self.label)
        layout.addWidget(self.slider)
        self.setLayout(layout)


class CrossSectionDisplay2D(object):
    def __init__(self, parent_widget, cross_section, color_label_prefix=""):
        self.parent_widget = parent_widget

        self.canvas = scene.SceneCanvas(keys='interactive', bgcolor='white')
        self.view = self.canvas.central_widget.add_view()
        # self.view.margin = 0
        # self.view.border_color = "white"
        # #self.view.border = (1, 0, 0, 1)
        self.view.camera.rect = (0, 0), (2, 2)
        self.cross_section = None
        self.slider = SliderWithLabel("")
        self.slider.slider.valueChanged.connect(self.set_L)
        self.cval_label = QtGui.QLabel()
        self.color_label_prefix = color_label_prefix
        self.color_value = None
        self.mesh = None
        self.view.on_mouse_press = self.on_mouse_press
        self.callbacks_right_click = []

        self.parent_widget.addWidget(self.canvas.native)
        self.parent_widget.addWidget(self.cval_label)
        self.parent_widget.addWidget(self.slider)

        self.draw_coordinate_axes()

        self.color_indicator = None
        self.color_value = [cross_section.L, 0, 0]

        self.set_cross_section(cross_section)
        self.update_color_value_label()
        self.draw_coordinate_axes()

    def on_mouse_press(self, event):
        if event.button == 2:
            # Event position (where the mouse click occurred, relative to the sub-plot window)
            x = event.pos[0]
            y = event.pos[1]

            # Minimum/maximum coordinates of the sub-plot
            xmin, ymin = self.view.pos
            xmax = xmin + self.view.rect.width
            ymax = ymin + self.view.rect.height

            # Normalised event coordinates (betwen 0 and 1). We need to
            # flip the y-coordinate because event coordinates run from top
            # to bottom but we need them to run from bottom to top.
            xn = x / (xmax - xmin)
            yn = (ymax - ymin - y) / (ymax - ymin)

            # Transform the normalised event coordinates into camera
            # coordinates (which are the "true" 2D coordinates that this
            # event corresponds to).
            pt_2d = np.array(self.view.camera.rect.pos) + np.array(self.view.camera.rect.size) * np.array([xn, yn])

            # Map the 2D coordinates back into 3D space and adjust the end
            # point of the line accordingly.
            pt_3d = self.cross_section.map_2d_to_3d(pt_2d)
            try:
                pt_RGB = lab2rgb(pt_3d, assert_valid=True)
            except RGBRangeError:
                # Do not adjust the line if the clicked point lies outside
                # the values which can be represented in RGB.
                return

            self.color_value = pt_3d
            self.update_color_value_label()
            self.redraw_color_indicator()

            for f in self.callbacks_right_click:
                f(event)

    def draw_coordinate_axes(self):
        self.axis_a = Line(color='gray', width=1)
        self.axis_b = Line(color='gray', width=1)
        self.axis_a.set_data(pos=np.array([[-110, 0], [120, 0]]))
        self.axis_b.set_data(pos=np.array([[0, -140], [0, 90]]))
        self.view.add(self.axis_a)
        self.view.add(self.axis_b)

    def redraw_color_indicator(self):
        if self.color_indicator != None:
            try:
                self.color_indicator.remove_parent(self.view.scene)
            except ValueError:
                pass

        # Only draw color indicator if we're in the correct cross section
        if self.cross_section.L == self.color_value[0]:
            self.color_indicator = Markers()
            self.color_indicator.set_data(pos=np.array(
                    [[self.color_value[1], self.color_value[2]]]))
            self.view.add(self.color_indicator)

    def update_color_value_label(self):
        val_lab = self.color_value
        val_rgb = lab2rgb(self.color_value)

        self.cval_label.setText(
            "{}L,a,b = {}  R,G,B = {}".format(
                self.color_label_prefix,
                "({:.0f}, {:.0f}, {:.0f})".format(val_lab[0], val_lab[1], val_lab[2]),
                "({:.1f}, {:.1f}, {:.1f})".format(val_rgb[0], val_rgb[1], val_rgb[2])))

    def add_callback_value_changed(self, f):
        self.slider.slider.valueChanged.connect(f)

    def add_callback_right_click(self, f):
        self.callbacks_right_click.append(f)

    def set_cross_section(self, cs):
        self.cross_section = cs
        self.set_L(cs.L)

    def set_L(self, L):
        self.cross_section.set_L(L)
        self.slider.slider.setValue(L)
        self.slider.label.setText("L={}".format(L))
        self.redraw()

    def redraw(self):
        if self.mesh is not None:
            self.mesh.remove_parent(self.view.scene)

        verts = self.cross_section.vertices_2d
        faces = self.cross_section.simplices
        vcolors = np.array([lab2rgb(pt) for pt in self.cross_section.vertices])
        self.view.camera.rect = (-110, -140), (230, 230)
        self.mesh = Mesh(vertices=verts, faces=faces, vertex_colors=vcolors)
        self.view.add(self.mesh)

        self.redraw_color_indicator()


def rgb2rgba(rgb):
    r, g, b = rgb
    return np.array([r, g, b, 1.])


class ColoredLine(object):
    def __init__(self, parent, N=100, N_markers=5):
        self.parent = parent
        self.N = N
        self.N_markers = N_markers
        self.line = Line()
        parent.add(self.line)
        self.markers = Markers()
        self.markers.set_style('o')
        self.parent.add(self.markers)

    def update(self, col1, col2):
        assert (col1 != None and col2 != None)
        col1 = np.asarray(col1, dtype=float)
        col2 = np.asarray(col2, dtype=float)
        self.pos = np.array([(1-t)*col1 + t*col2 for t in np.linspace(0., 1., self.N, endpoint=True)])
        self.colors = np.array([rgb2rgba(lab2rgb(pt, clip=True)) for pt in self.pos])
        self.line.set_data(pos=self.pos, color=self.colors, width=3)
        self.line.update()

        self.pos_markers = np.array([(1-t)*col1 + t*col2 for t in np.linspace(0., 1., self.N_markers, endpoint=True)])
        self.colors_markers = np.array([rgb2rgba(lab2rgb(pt, clip=True)) for pt in self.pos_markers])
        self.markers.set_data(self.pos_markers, face_color=self.colors_markers)



class CrossSectionDisplay3D(object):
    def __init__(self, parent_widget):
        self.parent_widget = parent_widget

        self.canvas = scene.SceneCanvas(keys='interactive', bgcolor='white')
        self.parent_widget.addWidget(self.canvas.native)
        self.view = self.canvas.central_widget.add_view()
        self.view.border_color = (0.5, 0.5, 0.5, 1)
        self.view.camera.rect = (-5, -5), (10, 10)
        self.view.border = (1, 0, 0, 1)
        self.view.set_camera('turntable', mode='perspective', up='z', distance=500,
                             azimuth=140., elevation=30.)
        self.line = ColoredLine(self.view)
        self.cross_sections = []
        self.meshes = []

        self.draw_boxed_coordinate_axis()

    def draw_boxed_coordinate_axis(self):
        Lmin, Lmax = 0, 100
        amin, amax = -110, 120
        bmin, bmax = -140, 90

        self.view.add(Line(pos=np.array([[Lmin, amin, bmin], [Lmax, amin, bmin]]), color='red', width=1))
        self.view.add(Line(pos=np.array([[Lmin, amax, bmin], [Lmax, amax, bmin]]), color='red', width=1))
        self.view.add(Line(pos=np.array([[Lmin, amax, bmax], [Lmax, amax, bmax]]), color='red', width=1))
        self.view.add(Line(pos=np.array([[Lmin, amin, bmax], [Lmax, amin, bmax]]), color='red', width=1))

        self.view.add(Line(pos=np.array([[Lmin, amin, bmin], [Lmin, amax, bmin]]), color='green', width=1))
        self.view.add(Line(pos=np.array([[Lmax, amin, bmin], [Lmax, amax, bmin]]), color='green', width=1))
        self.view.add(Line(pos=np.array([[Lmax, amin, bmax], [Lmax, amax, bmax]]), color='green', width=1))
        self.view.add(Line(pos=np.array([[Lmin, amin, bmax], [Lmin, amax, bmax]]), color='green', width=1))

        self.view.add(Line(pos=np.array([[Lmin, amin, bmin], [Lmin, amin, bmax]]), color='blue', width=1))
        self.view.add(Line(pos=np.array([[Lmin, amax, bmin], [Lmin, amax, bmax]]), color='blue', width=1))
        self.view.add(Line(pos=np.array([[Lmax, amax, bmin], [Lmax, amax, bmax]]), color='blue', width=1))
        self.view.add(Line(pos=np.array([[Lmax, amin, bmin], [Lmax, amin, bmax]]), color='blue', width=1))

        # # Add a 3D axis to keep us oriented
        # BoxedAxisVisual(0, 100, -110, 120, -140, 90, parent=self.view.scene)


    def add_cross_section(self, cs):
        self.cross_sections.append(cs)
        self.redraw_cross_sections()

    def remove_cross_section(self, cs):
        try:
            self.cross_sections.remove(cs)
        except ValueError:
            pass
        self.redraw_cross_sections()

    def redraw_cross_sections(self, *args):
        for mesh in self.meshes:
            try:
                mesh.remove_parent(self.view.scene)
            except ValueError:
                # This can happen during initialisation when the
                # meshes have not been added yet
                pass
        for cs in self.cross_sections:
            verts = cs.vertices
            #verts = np.roll(verts, 2, axis=1)  # Hack: Put L coordinate last so that L is plotted vertically
            faces = cs.simplices
            vcolors = np.empty((len(verts), 4))
            vcolors[:, 0:3] = np.array([lab2rgb(pt) for pt in verts])
            vcolors[:, 3] = 0.7

            mesh = scene.visuals.Mesh(vertices=verts, faces=faces, vertex_colors=vcolors)
            self.view.add(mesh)

            self.meshes.append(mesh)


class ColormapSelector(QtGui.QMainWindow):
    def __init__(self, sample_plot_functions):
        QtGui.QMainWindow.__init__(self)

        self.resize(700, 500)
        self.setWindowTitle('Colormap Selector')

        self.sample_plot_functions = sample_plot_functions

        cs1 = CrossSectionL(40)
        #cs2 = CrossSectionL(60)
        cs2 = CrossSectionL(90)

        # Central Widget
        self.splitter_h = QtGui.QSplitter(QtCore.Qt.Horizontal)
        self.splitter_v = QtGui.QSplitter(QtCore.Qt.Vertical)
        self.splitter_h.addWidget(self.splitter_v)
        self.setCentralWidget(self.splitter_h)

        self.cs_display_2d_L1 = CrossSectionDisplay2D(self.splitter_v, cs1, "Start color:  ")
        self.cs_display_2d_L2 = CrossSectionDisplay2D(self.splitter_v, cs2, "End color:    ")

        # 3D display for cross sections
        self.cs_display_3d = CrossSectionDisplay3D(self.splitter_h)
        self.cs_display_3d.add_cross_section(cs1)
        self.cs_display_3d.add_cross_section(cs2)
        self.cs_display_2d_L1.add_callback_value_changed(self.cs_display_3d.redraw_cross_sections)
        self.cs_display_2d_L2.add_callback_value_changed(self.cs_display_3d.redraw_cross_sections)

        self.cs_display_2d_L1.add_callback_right_click(self.update_color_line)
        self.cs_display_2d_L2.add_callback_right_click(self.update_color_line)

        self.cs_display_2d_L1.add_callback_right_click(self.update_matplotlib_plots)
        self.cs_display_2d_L2.add_callback_right_click(self.update_matplotlib_plots)

        # FPS message in statusbar:
        #self.status = self.statusBar()
        #self.status.showMessage("...")

        self.create_matplotlib_sample_figures()
        self.update_color_line()

    def update_color_line(self, *args):
        self.cs_display_3d.line.update(self.cs_display_2d_L1.color_value, self.cs_display_2d_L2.color_value)

    def create_matplotlib_sample_figures(self):
        """
        Create one matplotlib figure windows for each sample plot.
        """
        self.sample_plots = {}

        for f in self.sample_plot_functions:
            fig = plt.figure()
            ax = fig.gca()
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            self.sample_plots[f] = fig
        plt.show(block=False)
        self.update_matplotlib_plots()

    def update_matplotlib_plots(self, *args):
        for f in self.sample_plot_functions:
            fig = self.sample_plots[f]
            fig.clf()
            cmap = linear_colormap(self.cs_display_2d_L1.color_value, self.cs_display_2d_L2.color_value, coordspace='Lab')
            im = f(fig.gca(), cmap)
            cbar = fig.colorbar(im, drawedges=False)
            cbar.solids.set_edgecolor("face")
            fig.canvas.draw()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            # Close all matplotlib windows when the main GUI window is closed.
            plt.close('all')

            self.close()
