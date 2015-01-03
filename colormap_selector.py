import sys
import numpy as np
import dolfin as df
import matplotlib
matplotlib.use('wx')  # need a backend != PyQt to avoid conflicts with this GUI
import matplotlib.pyplot as plt
from PyQt4 import QtGui, QtCore
from vispy import app, scene, gloo
from vispy.scene.visuals import Line, Mesh, Markers
from vispy.visuals.transforms import STTransform
from helpers import transform_mesh, rgb2lab, lab2rgb, CrossSectionL, linear_colormap, RGBRangeError
from sample_plots import sample_scatterplot


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


N = 7
mesh = df.UnitCubeMesh(N, N, N)
mesh2 = transform_mesh(mesh, rgb2lab)


class MainWindow(QtGui.QMainWindow):
    sample_functions = [sample_scatterplot]

    def __init__(self):
        QtGui.QMainWindow.__init__(self)

        self.resize(700, 500)
        self.setWindowTitle('Colormap Selector')

        cs1 = CrossSectionL(mesh2, 40)
        #cs2 = CrossSectionL(mesh2, 60)
        cs2 = CrossSectionL(mesh2, 90)

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

        for f in self.sample_functions:
            fig = plt.figure()
            ax = fig.gca()
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            self.sample_plots[f] = fig
        plt.show(block=False)
        self.update_matplotlib_plots()

    def update_matplotlib_plots(self, *args):
        for f in self.sample_functions:
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



# Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    appQt = QtGui.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    appQt.exec_()
