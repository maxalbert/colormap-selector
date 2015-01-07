import sys
from PyQt4 import QtGui, QtCore
from cross_section import CrossSection, CrossSectionL, Plane
from cross_section_display import *


class MainWindow(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        self.resize(400, 400)
        self.setWindowTitle('CrossSectionDisplay3D Test')
        self.splitter_h = QtGui.QSplitter(QtCore.Qt.Horizontal)
        self.setCentralWidget(self.splitter_h)
        self.csd = CrossSectionDisplay3D()
        self.csd.add_to_widget(self.splitter_h)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()


def test_init_cross_section_display_3d():
    appQt = QtGui.QApplication(sys.argv)
    win = MainWindow()
    win.show()

    cs1 = CrossSectionL(L=10)
    cs2 = CrossSectionL(L=60)
    cs3 = CrossSectionL(L=80)
    cs4 = CrossSection(Plane(P=[50, 0, 0], n=[2, 4, -3]))

    win.csd.add_cross_section(cs1)
    win.csd.add_cross_section(cs2)
    win.csd.add_cross_section(cs3)
    win.csd.add_cross_section(cs4)

    win.csd.start_color = [40, 70, -100]
    win.csd.end_color = [80, -15, 70]

    # Here we change L *after* the cross section has been added to the
    # CrossSectionDisplay. This is to check that redraw() takes this
    # change into account.
    cs1.L = 40
    win.csd.redraw()

    appQt.exec_()


if __name__ == '__main__':
    test_init_cross_section_display_3d()
