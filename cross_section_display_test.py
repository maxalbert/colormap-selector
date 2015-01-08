import sys
from PyQt4 import QtGui, QtCore
from cross_section import CrossSection, CrossSectionL, Plane
from cross_section_display import *


class MainWindow(QtGui.QMainWindow):
    def __init__(self, cross_section_display):
        QtGui.QMainWindow.__init__(self)
        self.resize(400, 400)
        self.setWindowTitle('CrossSectionDisplay3D Test')
        self.splitter_h = QtGui.QSplitter(QtCore.Qt.Horizontal)
        self.setCentralWidget(self.splitter_h)
        self.csd = cross_section_display
        self.csd.add_to_widget(self.splitter_h)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()


def test_init_cross_section_display_2d():
    appQt = QtGui.QApplication(sys.argv)

    cs = CrossSectionL(L=10)
    csd = CrossSectionDisplay2D(cs)

    win = MainWindow(csd)
    win.show()

    # Here we change L *after* the cross section has been added to the
    # CrossSectionDisplay. This is to check that redraw() takes this
    # change into account.
    cs.L = 40
    csd.redraw()

    appQt.exec_()


def test_init_cross_section_display_2d_const_L():
    appQt = QtGui.QApplication(sys.argv)

    csd = CrossSectionDisplay2DConstL(L=10, color_label_prefix="Selected color: ")

    win = MainWindow(csd)
    win.show()

    # Here we change L *after* the cross section has been added to the
    # CrossSectionDisplay. This is to check that redraw() takes this
    # change into account.
    csd.set_L(40)
    csd.selected_color = [40, 10, 10]
    csd.redraw()

    appQt.exec_()


def test_init_cross_section_display_3d():
    appQt = QtGui.QApplication(sys.argv)

    cs1 = CrossSectionL(L=10)
    cs2 = CrossSectionL(L=60)
    cs3 = CrossSectionL(L=80)
    cs4 = CrossSection(Plane(P=[50, 0, 0], n=[2, 4, -3]))

    csd = CrossSectionDisplay3D()
    csd.add_cross_section(cs1)
    csd.add_cross_section(cs2)
    csd.add_cross_section(cs3)
    csd.add_cross_section(cs4)

    csd.start_color = [40, 70, -100]
    csd.end_color = [80, -15, 70]

    win = MainWindow(csd)
    win.show()

    # Here we change L *after* the cross section has been added to the
    # CrossSectionDisplay. This is to check that redraw() takes this
    # change into account.
    cs1.L = 40
    csd.redraw()

    appQt.exec_()


if __name__ == '__main__':
    #test_init_cross_section_display_2d()
    test_init_cross_section_display_2d_const_L()
    #test_init_cross_section_display_3d()
