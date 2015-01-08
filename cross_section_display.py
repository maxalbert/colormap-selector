import matplotlib
matplotlib.use('wx')
import matplotlib.pyplot as plt
import numbers
import numpy as np
from PyQt4 import QtGui, QtCore
from vispy import scene
from vispy.scene.visuals import Mesh, Line, Markers
from color_transformations import lab2rgb, lab2rgba, RGBRangeError, linear_colormap
from cross_section import CrossSectionL


class SliderWithLabel(QtGui.QWidget):
    def __init__(self, label="", *args, **kwargs):
        QtGui.QWidget.__init__(self, *args, **kwargs)
        self.slider = QtGui.QSlider()
        self.slider.setOrientation(QtCore.Qt.Horizontal)
        self.slider.setRange(1, 99)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(10)
        self.label = QtGui.QLabel(self)
        self.label.setText(label)
        layout = QtGui.QHBoxLayout()
        layout.setMargin(0)
        layout.setSpacing(2)
        layout.addWidget(self.label)
        layout.addWidget(self.slider)
        self.setLayout(layout)


class CrossSectionDisplay2D(object):
    def __init__(self, cross_section, color_label_prefix=""):
        self.canvas = scene.SceneCanvas(keys='interactive', bgcolor='white')
        self.view = self.canvas.central_widget.add_view()
        self.view.camera.rect = (-110, -140), (230, 230)
        self.view.on_mouse_press = self.on_mouse_press
        self.callbacks_right_click = []

        self.cross_section = cross_section

        self.color_label_prefix = color_label_prefix
        self.selected_color = self.cross_section.plane.pt
        self.color_indicator = Markers()
        self.view.add(self.color_indicator)

        self.cs_mesh = Mesh()
        self.view.add(self.cs_mesh)

        self.color_value_label = QtGui.QLabel()
        self.splitter_v = QtGui.QSplitter(QtCore.Qt.Vertical)
        self.splitter_v.addWidget(self.canvas.native)
        self.splitter_v.addWidget(self.color_value_label)

        self.redraw()

    def add_callback_right_click(self, f):
        self.callbacks_right_click.append(f)

    def add_to_widget(self, parent_widget):
        self.parent_widget = parent_widget
        self.parent_widget.addWidget(self.splitter_v)

    def redraw(self):
        self.update_color_value_label()
        self.update_color_indicator()
        self.cs_mesh.set_data(vertices=self.cross_section.vertices_2d,
                              faces=self.cross_section.faces,
                              vertex_colors=self.cross_section.vertex_colors)

    def set_L(self, L):
        self.cross_section.L = L

    def update_color_value_label(self):
        assert self.selected_color != None

        val_lab = self.selected_color
        val_rgb = lab2rgb(self.selected_color)

        self.color_value_label.setText(
            "{}L,a,b = {}  R,G,B = {}".format(
                self.color_label_prefix,
                "({:.0f}, {:.0f}, {:.0f})".format(val_lab[0], val_lab[1], val_lab[2]),
                "({:.1f}, {:.1f}, {:.1f})".format(val_rgb[0], val_rgb[1], val_rgb[2])))

    def transform_event_to_color_coordinates(self, pos):
        # Event position (where the mouse click occurred, relative to the sub-plot window)
        x, y = pos

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
        pt_2d = (np.array(self.view.camera.rect.pos) +
                 np.array(self.view.camera.rect.size) * np.array([xn, yn]))

        return pt_2d

    def on_mouse_press(self, event):
        if event.button == 2:
            pt_2d = self.transform_event_to_color_coordinates(event.pos[:2])
            pt_lab = self.cross_section.mapping_3d_to_2d.apply_inv(pt_2d)

            try:
                _ = lab2rgb(pt_lab, assert_valid=True)
            except RGBRangeError:
                # Do not adjust the line if the clicked point lies outside
                # the values which can be represented in RGB.
                return

            self.selected_color = pt_lab
            self.redraw()

            for f in self.callbacks_right_click:
                f(event)

    def update_color_indicator(self):
        if self.selected_color is not None:
            pos = self.cross_section.mapping_3d_to_2d.apply(self.selected_color)
            self.color_indicator.set_data(pos=pos.reshape(1, 2))


class CrossSectionDisplay2DConstL(CrossSectionDisplay2D):
    def __init__(self, arg, color_label_prefix=""):
        if isinstance(arg, CrossSectionL):
            cs = arg
        elif isinstance(arg, numbers.Number):
            cs = CrossSectionL(L=arg)
        else:
            raise TypeError("CrossSectionDisplay2DConstL must be initialised "
                            "with a number or an object of type CrossSectionL. "
                            "Got: {} (type: {}).".format(arg, type(arg)))
        super(CrossSectionDisplay2DConstL, self).__init__(cs, color_label_prefix)
        self.sliderlabel = SliderWithLabel()
        self.splitter_v.addWidget(self.sliderlabel)
        self.set_L(cs.L)
        self.sliderlabel.slider.valueChanged.connect(self.set_L)

    def add_to_widget(self, parent_widget):
        self.parent_widget = parent_widget
        self.parent_widget.addWidget(self.splitter_v)

    def set_L(self, L):
        self.cross_section.L = L
        self.sliderlabel.slider.setValue(L)
        self.sliderlabel.label.setText("L={}".format(L))
        self.redraw()

    def add_callback_slider_value_changed(self, f):
        self.sliderlabel.slider.valueChanged.connect(f)


class ColoredLine3D(object):
    def __init__(self, parent):
        self.parent = parent
        self.line = Line()
        self.markers = Markers()
        self.markers.set_style('o')
        self.parent.add(self.line)
        self.parent.add(self.markers)

    def update(self, col1, col2, N_markers=5):
        assert (col1 != None and col2 != None)

        col1 = np.asarray(col1, dtype=float)
        col2 = np.asarray(col2, dtype=float)

        pos = np.array([(1-t)*col1 + t*col2 for t in np.linspace(0., 1., 100, endpoint=True)])
        colors = np.array([lab2rgba(pt, clip=True) for pt in pos])
        self.line.set_data(pos=pos, color=colors, width=3)
        self.line.update()

        pos_markers = np.array([(1-t)*col1 + t*col2 for t in np.linspace(0., 1., N_markers, endpoint=True)])
        colors_markers = np.array([lab2rgba(pt, clip=True) for pt in pos_markers])
        self.markers.set_data(pos_markers, face_color=colors_markers)


class CrossSectionDisplay3D(object):
    def __init__(self):
        self.cs_meshes = {}
        self.start_color = None
        self.end_color = None
        self.canvas = scene.SceneCanvas(keys='interactive', bgcolor='white')
        self.view = self.canvas.central_widget.add_view()
        self.view.border_color = (0.5, 0.5, 0.5, 1)
        self.view.camera.rect = (-5, -5), (10, 10)
        self.view.border = (1, 0, 0, 1)
        self.view.set_camera('turntable', mode='perspective', up='z',
                             azimuth=140., elevation=30.,  distance=500)
        self.colored_line = None
        self.redraw()

    def add_to_widget(self, parent_widget):
        self.parent_widget = parent_widget
        self.parent_widget.addWidget(self.canvas.native)

    def add_cross_section(self, cs):
        vertex_colors = cs.vertex_colors.copy()
        vertex_colors[:, 3] = 0.7  # smaller alpha value for slight transparency
        mesh = Mesh(vertices=cs.vertices_3d, faces=cs.faces, vertex_colors=vertex_colors)
        self.view.add(mesh)
        self.cs_meshes[cs] = mesh

    def redraw(self):
        self.draw_boxed_coordinate_axes()

        for cs, mesh in self.cs_meshes.iteritems():
            vertex_colors = cs.vertex_colors.copy()
            vertex_colors[:, 3] = 0.7  # smaller alpha value for slight transparency
            mesh.set_data(vertices=cs.vertices_3d, faces=cs.faces, vertex_colors=vertex_colors)

        self.draw_colored_line()

    def draw_colored_line(self, N_markers=5):
        """
        Draw a colored line connecting the start and end color.
        The argument `N` specifies how many intermediate colored
        points to draw.
        """
        if self.start_color is None or self.end_color is None:
            return

        if self.colored_line is None:
            self.colored_line = ColoredLine3D(self.view)

        self.colored_line.update(self.start_color, self.end_color)

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


class ColormapSelector(QtGui.QMainWindow):
    def __init__(self, sample_plot_functions):
        QtGui.QMainWindow.__init__(self)

        self.resize(700, 500)
        self.setWindowTitle('Colormap Selector')

        self.sample_plot_functions = sample_plot_functions

        cs1 = CrossSectionL(40)
        cs2 = CrossSectionL(90)

        self.splitter_h = QtGui.QSplitter(QtCore.Qt.Horizontal)
        self.setCentralWidget(self.splitter_h)
        self.splitter_v = QtGui.QSplitter(QtCore.Qt.Vertical)
        self.splitter_h.addWidget(self.splitter_v)

        self.cs_display_2d_L1 = CrossSectionDisplay2DConstL(cs1, "Start color:  ")
        self.cs_display_2d_L2 = CrossSectionDisplay2DConstL(cs2, "End color:    ")
        self.cs_display_2d_L1.add_to_widget(self.splitter_v)
        self.cs_display_2d_L2.add_to_widget(self.splitter_v)

        self.cs_display_3d = CrossSectionDisplay3D()
        self.cs_display_3d.add_cross_section(cs1)
        self.cs_display_3d.add_cross_section(cs2)
        self.cs_display_3d.start_color = self.cs_display_2d_L1.selected_color
        self.cs_display_3d.end_color = self.cs_display_2d_L2.selected_color
        self.cs_display_3d.add_to_widget(self.splitter_h)
        self.cs_display_3d.redraw()

        self.cs_display_2d_L1.add_callback_right_click(self.set_start_color)
        self.cs_display_2d_L2.add_callback_right_click(self.set_end_color)
        self.cs_display_2d_L1.add_callback_right_click(self.update_matplotlib_plots)
        self.cs_display_2d_L2.add_callback_right_click(self.update_matplotlib_plots)
        self.cs_display_2d_L1.add_callback_slider_value_changed(self.cs_display_3d.redraw)
        self.cs_display_2d_L2.add_callback_slider_value_changed(self.cs_display_3d.redraw)

        self.create_matplotlib_sample_figures()

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
            cmap = linear_colormap(self.cs_display_2d_L1.selected_color,
                                   self.cs_display_2d_L2.selected_color,
                                   coordspace='Lab')
            im = f(fig.gca(), cmap)
            cbar = fig.colorbar(im, drawedges=False)
            cbar.solids.set_edgecolor("face")
            fig.canvas.draw()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            plt.close('all')  # close all matplotlib windows
            self.close()      # close the main GUI window

    def set_start_color(self, event):
        self.cs_display_3d.start_color = self.cs_display_2d_L1.selected_color
        self.cs_display_3d.redraw()

    def set_end_color(self, event):
        self.cs_display_3d.end_color = self.cs_display_2d_L2.selected_color
        self.cs_display_3d.redraw()
