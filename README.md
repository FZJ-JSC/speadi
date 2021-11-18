<h1 class="title"> MDEnvironment <br /> <span class="subtitle"> A Python package that aims to characterise the dynamics of local chemical environments from Molecular Dynamics trajectories of proteins and other biomolecules. </span> </h1>

<a href="https://gitlab.jsc.fz-juelich.de/debruyn1/mdenvironment/-/commits/master"><img alt="pipeline status" src="https://gitlab.jsc.fz-juelich.de/debruyn1/mdenvironment/badges/master/pipeline.svg" /></a>  <a href="https://gitlab.jsc.fz-juelich.de/debruyn1/mdenvironment/-/commits/master"><img alt="coverage report" src="https://gitlab.jsc.fz-juelich.de/debruyn1/mdenvironment/badges/master/coverage.svg" /></a>

# Introduction

# Quick install

For installation into the default python environment, run the following in a terminal:

```bash
pip install --user git+https://pip_token:F4ZJGgGyyb62dP34xMqo@gitlab.jsc.fz-juelich.de/debruyn1/mdenvironment.git
```

# Time-resolved RDFs

This package provides an optimised function to calculate Time-resolved Radial Distribution Functions using atomistic simulation trajectory data.

Trajectory data may be anything that the package `MDTraj` can handle, or preferably a string pointing to the location of such data.

## Explanation

Normally, Radial Distribution Functions used in atomistic simulations are averaged over whole trajectories. `MDEnvironment` averages over user-defined windows of time. This gives a separate RDF between group *a* and *b* for each window in the trajectory.

<img src="docs/trrdf.svg" width="850px">

# Van Hove functions

# Installation

Installation is provided easily through `pip`. It can be installed either directly as a package, or as an editable source.

## Direct installation

For installation into the default python environment, run the following in a terminal:

```bash
pip install git+https://gitlab.jsc.fz-juelich.de/debruyn1/mdenvironment
```

To install for just the current user, add the `--user` option:

```bash
pip install --user git+https://gitlab.jsc.fz-juelich.de/debruyn1/mdenvironment
```

## Editable source installation

Open up a terminal. Navigate to the location you want to clone this repository. Then, run the following to clone the whole repository:

```bash
git clone ssh://git@gitlab.jsc.fz-juelich.de:10022/debruyn1/mdenvironment
```

Then, install locally using `pip` (after entering the package sub-directory):

```bash
cd mdenvironment/ && pip install -e .
```

## [WIP] Usage

To calculate the time-resolved RDF for every single protein heavy atom with each ion species in solvent, you first need to specify the trajectory and topology to be used:

```python
topology = './topology.gro'
trajectory = './trajectory.xtc'
```

Next, load the topology in `MDTraj` and subset into useful groups:

```python
import mdtraj as md

top = md.load_topology(topology)
na = top.select('name NA')
cl = top.select('name CL')
protein_by_atom = [top.select(f'index {ix}') for ix in top.select('protein and not type H')]
```

Now you can load `time-resolved RDF` to analyse the RDFs:

```python
from mylibrary import grt, plot_grt, plot_map
```

To make an RDF for each heavy protein atom

```python
r, g_rt = grt(trajectory, protein_by_atom, [na, cl], top=top, n_windows=4_500, window_size=100,\
              skip=0, opt=True, pbc='ortho', stride=1, nbins=10)
```

To repeat the analysis, but obtain un-normed raw histograms of distances instead, set the key `raw_counts` to `True`.

```python
r, g_rt = grt(trajectory, protein_by_atom, [na, cl], top=top, n_windows=4_500, window_size=100,\
              skip=0, opt=True, pbc='ortho', stride=1, nbins=10, raw_counts=True)
```

### To-do

-   add examples of the plotting function in action

## [WIP] Citation

Add Zenodo link as soon as a first public release is planned to coincide with open-sourcing.

## Contributors

-   Emile de Bruyn

## [WIP] License

### To-dos

-   add LGPL license
-   check with colleagues and legal department before publication
