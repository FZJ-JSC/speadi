"""
Used by plotting functions to display the time evolution of radial
distributions.

Adapted from https://stackoverflow.com/questions/38208700/matplotlib-plot-lines-with-colors-through-colormap
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


def multiline(xs, ys, c, ax=None, **kwargs):
    """
    Plot lines with continuous colours

    Parameters
    ----------
    xs : list, numpy.array
    	Iterable container of x coordinates
    ys : list, numpy.array
    	Iterable container of y coordinates
    c : list, numpy.array
    	Iterable container of numbers mapped to colormap

    Other parameters
    ----------------
    ax : matplotlib.Axes
    	(Optional) axes instance to plot on, else create a new axes instance
    kwargs : dict
    	Dictionary of keyword arguments passed to matplotlib.LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed
    	by i)

    Returns
    -------
    lc : matplotlib.LineCollection
    	LineCollection that can be added to an existing plot canvas
    """

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()

    return lc
