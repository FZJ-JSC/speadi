- [Introduction](#sec-1)
- [Installation](#sec-2)
  - [Direct installation](#sec-2-1)
  - [Editable source installation](#sec-2-2)
  - [[WIP] Usage](#sec-2-3)
    - [To-do](#sec-2-3-1)
  - [[WIP] Citation](#sec-2-4)
  - [Contributors](#sec-2-5)
  - [[WIP] License](#sec-2-6)
    - [To-dos](#sec-2-6-1)
- [Van Hove functions](#sec-3)
- [Time-resolved RDFs](#sec-4)
  - [Explanation](#sec-4-1)


# Introduction<a id="sec-1"></a>

# Installation<a id="sec-2"></a>

Installation is provided easily through `pip`. It can be installed either directly as a package, or as an editable source.

## Direct installation<a id="sec-2-1"></a>

For installation into the default python environment, run the following in a terminal:

```bash
pip install git+https://gitlab.jsc.fz-juelich.de/debruyn1/mdenvironment
```

To install for just the current user, add the `--user` option:

```bash
pip install --user git+https://gitlab.jsc.fz-juelich.de/debruyn1/mdenvironment
```

## Editable source installation<a id="sec-2-2"></a>

Open up a terminal. Navigate to the location you want to clone this repository. Then, run the following to clone the whole repository:

```bash
git clone ssh://git@gitlab.jsc.fz-juelich.de:10022/debruyn1/mdenvironment
```

Then, install locally using `pip` (after entering the package sub-directory):

```bash
cd mdenvironment/ && pip install -e .
```

## [WIP] Usage<a id="sec-2-3"></a>

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

### To-do<a id="sec-2-3-1"></a>

-   add examples of the plotting function in action

## [WIP] Citation<a id="sec-2-4"></a>

Add Zenodo link as soon as a first public release is planned to coincide with open-sourcing.

## Contributors<a id="sec-2-5"></a>

-   Emile de Bruyn

## [WIP] License<a id="sec-2-6"></a>

### To-dos<a id="sec-2-6-1"></a>

-   add LGPL license
-   check with colleagues and legal department before publication

# Van Hove functions<a id="sec-3"></a>

# Time-resolved RDFs<a id="sec-4"></a>

This package provides an optimised function to calculate Time-resolved Radial Distribution Functions using atomistic simulation trajectory data.

Trajectory data may be anything that the package `MDTraj` can handle, or preferably a string pointing to the location of such data.

## Explanation<a id="sec-4-1"></a>

Normally, Radial Distribution Functions used in atomistic simulations are averaged over whole trajectories. `MDEnvironment` averages over user-defined windows of time. This gives a separate RDF between group *a* and *b* for each window in the trajectory.

<img src="docs/trrdf.svg" width="850px">
