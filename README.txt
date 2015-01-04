Quick start
===========

- Make sure that you have all dependencies installed (see below).

- To start the GUI, run

     python colormap_selector.py

- Right-click into the two cross section plots on the left to change
  the start/end color of the colormap.

- Use the sliders to select a different value of ``L`` (= lightness) for
  each cross section.

- To quit, press the "Esc" key or close the main GUI window.

Whenever the colormap changes the 3D view on the right shows a linear
interpolation of the start/end colors in ``CIELab`` space. Moreover,
the sample scatterplot (in the separate window) is adjusted to use the
new colormap.


Dependencies
============

Currently the following dependencies are needed. Hopefully some of
these can be eliminated in the future.

- Vispy (recent development version > 0.3.0)

- Matplotlib

- PyQt

- wxpython

`IPython notebook <http://ipython.org/notebook.html>`__
- FEniCS (see `installation instructions <http://fenicsproject.org/download/>`__); if possible, it is highly
  recommended to use one of the pre-packaged versions, e.g. for Ubuntu.


Background Info
===============

This is a GUI to facilitate the design of colormaps with "good"
properties. It originated from a discussion `discussion <http://sourceforge.net/p/matplotlib/mailman/matplotlib-devel/?viewmonth=201411&viewday=21&style=threaded>`__ on the matplotlib
mailing list about designing a new default colormap.

One of the major problems with many existing colormaps (including the
infamous and harmful rainbow/jet colormaps), is that they are not
perceptionally linear. This means that equal changes in the values of
the displayed data do not correspond to color changes that are
perceived as equal by humans, which is particularly harmful for
scientific applications. For details see the references below.

The most commonly used `RGB color space <http://en.wikipedia.org/wiki/RGB_color_space>`__ is not well suited for
colormap design. Instead, there are other color spaces which are
specifically designed to be perceptually linear. One example is the
`CIELab color space <http://en.wikipedia.org/wiki/Lab_color_space>`__ which represents each color using three
the parameters ``L`` (= lightness), ``a`` (= red/green component) and ``b``
(= yellow/blue component).

This GUI helps defining perceptually linear colormaps. It consists of
one main window which on the left displays two cross sections through
``CIELab`` space. Each of these corresponds to all those colors with
a fixed value of ``L`` (lightness) that are representable in ``RGB`` space.
The ``L``-value for each cross section can be changed using the slider
underneath.

The start/end color of the colormap can be selected by right-clicking
inside the cross sections. The two selected colors are then interpolated
linearly and this defines the whole colormap. The 3D view on the right
illustrates this linear interpolation.

In addition, a sample scatterplot is displayed in a separate window
which is adjusted whenever the colormap changes.


References
==========

A random (and highly incomplete) selection of articles on color, colormaps and scientific visualization:

- Subtleties of color (Part 1 of 6)
  http://earthobservatory.nasa.gov/blogs/elegantfigures/2013/08/05/subtleties-of-color-part-1-of-6/

- Matplotlib: choosing colormaps
  http://matplotlib.org/users/colormaps.html

- Mycarta ("A blog about Geophysics, Visualization, Image Processing, and Planetary Science"), Category: Visualization
  http://mycarta.wordpress.com/category/visualization/

- Rainbow Colormaps – What are they good for? Absolutely nothing!
  http://medvis.org/2012/08/21/rainbow-colormaps-what-are-they-good-for-absolutely-nothing/

- Why Should Engineers and Scientists Be Worried About Color?
  http://www.research.ibm.com/people/l/lloydt/color/color.HTM
