import numpy as np


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


class Mapping3Dto2D(object):
    def __init__(self, plane):
        self.plane = plane
        self.idx_const = None
        self.idcs_nonconst = None
        self.mat = None
        self.mat_inv = None
        self.compute_matrices()

    def compute_matrices(self):
        R = self.compute_rotation_matrix()  # rotation matrix
        T = np.eye(3)  # translation matrix; TODO: need to determine this from the points to be mapped (currently always the identity matrix)
        self.idcs_nonconst = [k for k in xrange(3) if k != self.idx_const]
        P = np.eye(3)[self.idcs_nonconst]  # projection matrix
        self.mat = np.dot(P, np.dot(T, R))

        self.const_coord = np.dot(R, self.plane.pt)[self.idx_const]
        self.mat_inv = np.dot(R, T)

    def compute_rotation_matrix(self):
        v = self.plane.n  # normal vector of the plane

        # Find the coordinate axis `w` which has the smallest angle
        # with respect to the normal vector of the plane: it is the
        # axis whose dot product with v is largest, i.e. which v has
        # its maximum coordinate.
        self.idx_const = np.argmax(np.absolute(v))
        w = np.array([0., 0., 0.])
        w[self.idx_const] = 1.0

        # We rotate the points into a plane with constant idx-coordinates,
        # i.e. a plane orthogonal to the vector w. The rotation axis is
        # given by the cross product of v and w and the angle by their dot
        # product (after normalising).
        rot_axis = np.cross(v, w)
        if np.linalg.norm(rot_axis) < 1e-8:
            rot_axis = np.array([1, 0, 0])
            theta = 0.0
        else:
            rot_axis /= np.linalg.norm(rot_axis)
            theta = np.arccos(np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w)))  # rotation angle
        R = rotation_matrix(rot_axis, theta)
        #pts_rotated = np.array([np.dot(R, pt) for pt in pts])
        return R

    def apply(self, pts_3d):
        """
        Apply the 3d->2d transformation to the given points `pts_3d`.

        *Arguments*
        """
        pts_3d = np.asarray(pts_3d, dtype=float)

        # Ensure that `pts_3d` is a list of 3D points
        if pts_3d.ndim == 1:
            pts_3d.shape = (1, len(pts_3d))
        assert pts_3d.ndim == 2 and pts_3d.shape[1] == 3

        pts_2d = np.dot(self.mat, pts_3d.T).T

        return pts_2d

    def apply_inv(self, pts_2d):
        """
        Apply the 2d->3d transformation to the given points `pts`.

        *Arguments*
        """
        pts_2d = np.asarray(pts_2d, dtype=float)

        # Ensure that `pts` is a list of 3D points
        if pts_2d.ndim == 1:
            pts_2d.shape = (1, len(pts_2d))
        assert pts_2d.ndim == 2 and pts_2d.shape[1] == 2

        pts_2d_embedded = np.empty((len(pts_2d), 3))
        pts_2d_embedded[:, self.idcs_nonconst] = pts_2d
        pts_2d_embedded[:, self.idx_const] = self.const_coord

        pts_3d = np.dot(self.mat_inv, pts_2d_embedded.T).T

        return pts_3d
