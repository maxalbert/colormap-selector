import matplotlib
matplotlib.use('wx')  # need a backend != PyQt to avoid conflicts with this GUI
from helpers import *
from sample_plots import sample_scatterplot

sample_plot_functions = [sample_scatterplot]


# Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    appQt = QtGui.QApplication(sys.argv)
    win = ColormapSelector(sample_plot_functions)
    win.show()
    appQt.exec_()
