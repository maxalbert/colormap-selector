import sys
import matplotlib
matplotlib.use('Qt4Agg')
from PyQt4 import QtGui
from cross_section_display import ColormapSelector
from sample_plots import sample_scatterplot

sample_plot_functions = [sample_scatterplot]


# Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    appQt = QtGui.QApplication(sys.argv)
    win = ColormapSelector(sample_plot_functions)
    win.show()
    appQt.exec_()
