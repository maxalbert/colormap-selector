import numpy as np
import dolfin as df
import matplotlib.colors as mcolors
from scipy.spatial import Delaunay

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

    return L, a, b


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


def cross_section_mesh(mesh_3d, mesh_2d):
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


class CrossSection(object):
    def __init__(self, mesh_3d, mesh_2d):
        self.vertices, self.simplices = cross_section_mesh(mesh_3d, mesh_2d)
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
    def __init__(self, mesh_3d, L):
        self.mesh_3d = mesh_3d
        self.set_L(L)

    def set_L(self, L):
        self.L = L
        coords = self.mesh_3d.coordinates()
        amin, amax = coords[:, 1].min(), coords[:, 1].max()
        bmin, bmax = coords[:, 2].min(), coords[:, 2].max()
        A = [L, amin, bmin]
        B = [L, amax, bmin]
        C = [L, amax, bmax]
        D = [L, amin, bmax]
        mesh_square = create_quadrilateral_mesh(A, B, C, D)
        self.vertices, self.simplices = cross_section_mesh(self.mesh_3d, mesh_square)
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
