from PyQt4 import QtGui, QtCore
from vispy import scene
from vispy.scene.visuals import Mesh, Line
from color_transformations import lab2rgb
import numpy as np


class CrossSectionDisplay3D(object):
    def __init__(self):
        self.cs_meshes = {}
        self.canvas = scene.SceneCanvas(keys='interactive', bgcolor='white')
        self.view = self.canvas.central_widget.add_view()
        self.view.border_color = (0.5, 0.5, 0.5, 1)
        self.view.camera.rect = (-5, -5), (10, 10)
        self.view.border = (1, 0, 0, 1)
        self.view.set_camera('turntable', mode='perspective', up='z',
                             azimuth=140., elevation=30.,  distance=500)
        self.redraw()

    def add_to_widget(self, parent_widget):
        self.parent_widget = parent_widget
        self.parent_widget.addWidget(self.canvas.native)

    def add_cross_section(self, cs):
        mesh = Mesh(vertices=cs.vertices_3d, faces=cs.faces, vertex_colors=cs.vertex_colors)
        self.view.add(mesh)
        self.cs_meshes[cs] = mesh

    def redraw(self):
        self.draw_boxed_coordinate_axes()
        for cs, mesh in self.cs_meshes.iteritems():
            mesh.set_data(vertices=cs.vertices_3d, faces=cs.faces, vertex_colors=cs.vertex_colors)

    def draw_boxed_coordinate_axes(self):
        Lmin, Lmax = 0, 100
        amin, amax = -110, 120
        bmin, bmax = -140, 90

        box_x = np.array([[Lmin, amin, bmin], [Lmax, amin, bmin],
                          [Lmin, amax, bmin], [Lmax, amax, bmin],
                          [Lmin, amax, bmax], [Lmax, amax, bmax],
                          [Lmin, amin, bmax], [Lmax, amin, bmax]])

        box_y = np.array([[Lmin, amin, bmin], [Lmin, amax, bmin],
                          [Lmax, amin, bmin], [Lmax, amax, bmin],
                          [Lmax, amin, bmax], [Lmax, amax, bmax],
                          [Lmin, amin, bmax], [Lmin, amax, bmax]])

        box_z = np.array([[Lmin, amin, bmin], [Lmin, amin, bmax],
                          [Lmin, amax, bmin], [Lmin, amax, bmax],
                          [Lmax, amax, bmin], [Lmax, amax, bmax],
                          [Lmax, amin, bmin], [Lmax, amin, bmax]])

        self.view.add(Line(pos=box_x, color='red', connect='segments', width=1))
        self.view.add(Line(pos=box_y, color='green', connect='segments', width=1))
        self.view.add(Line(pos=box_z, color='blue', connect='segments', width=1))
