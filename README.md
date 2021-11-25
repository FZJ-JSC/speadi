<h1 class="title"> MDEnvironment <br /> </h1>

A Python package that aims to characterise the dynamics of local chemical environments from Molecular Dynamics trajectories of proteins and other biomolecules.

<a href="https://gitlab.jsc.fz-juelich.de/debruyn1/mdenvironment/-/commits/master"><img alt="pipeline status" src="https://gitlab.jsc.fz-juelich.de/debruyn1/mdenvironment/badges/master/pipeline.svg" /></a>  <a href="https://gitlab.jsc.fz-juelich.de/debruyn1/mdenvironment/-/commits/master"><img alt="coverage report" src="https://gitlab.jsc.fz-juelich.de/debruyn1/mdenvironment/badges/master/coverage.svg" /></a>

# Introduction

MDEnvironment provides the user tools with which to characterise the local chemical environment using Molecular Dynamics data. At the moment, it implements two variations of the pair radial distribution function (RDF): time-resolved RDFs (TRRDFs) and van Hove dynamic correlation functions (VHFs). Selecting surface or active 'patches' of atoms in biomolecules is provided as a rough tool at the moment, but the aim of the package is to provide this in a formalised way that complements the analysis using TRRDFs and VHFs.

# Quick install

For installation into the default python environment, run the following in a terminal:

```bash
pip install --user git+https://pip_token:F4ZJGgGyyb62dP34xMqo@gitlab.jsc.fz-juelich.de/debruyn1/mdenvironment.git
```

# RDFs

Normally, Radial Distribution Functions used in atomistic simulations are averaged over entire trajectories. `MDEnvironment` averages over user-defined windows of time. This gives a separate RDF between group *a* and *b* for each window in the trajectory.

# Time-resolved RDFs

This package provides a method to calculate Time-resolved Radial Distribution Functions using atomistic simulation trajectory data. This is implemented efficiently using `Numpy` arrays and the `Numba` package when available. Trajectory data may be anything that the package `MDTraj` can handle, or preferably a string pointing to the location of such data.

## Data requirements

The trajectory data used must be sampled to a sufficient degree. The sampling frequency of the input data naturally determines the time-resolving capability of the time-resolved RDF. When using both small groups for reference particles and selection particles, the sampling frequency must be high enough to provide an ensemble of distances in each time slice that provides a satisfactory signal-to-noise ratio.

In RDFs of particular ions around selected single atoms in all-atom simulations of biomolecules, experience suggests a sampling frequency of below 1 picosecond and window lengths of 1-10 picoseconds to follow changes in coordination shells of atoms.

# Van Hove functions

This package also provides a method to calculate the van Hove dynamic correlation function using atomistic simulation trajectory data. This is also implemented efficiently using `Numpy` arrays and the `Numba` package when available. Trajectory data may be anything that the package `MDTraj` can handle, or preferably a string pointing to the location of such data.

## Data requirements

As with TRRDFs, the input trajectory data must be sampled above a certain frequency. Window lengths of 1-2 picoseconds are enough to follow the loss in structure in all-atom simulations of water, with each window containing anything above 10 samples.

VHFs of ions around single atoms in do not require sample frequencies above those consistent with the time-scale of the movement of ions, yet they do require a larger number of windows to average over. This number of windows can either be supplied by using a trajectory or trajectory slice of preferably at least 10 ns in length, or alternatively by using a sliding window over a trajectory slice of at least 1 ns in length.

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

Open up a terminal. Navigate to the location you want to clone this repository. Then, run the following to clone the entire repository:

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

## To-Do

-   [ ] generate plots for documentation as showcase and teaser
-   [X] implement matrix based calculation of MIC convention for general PBC
-   [ ] re-write MIC convention for general PBC in for-loops for `Numba`
-   [ ] add defaults to docstrings
-   [ ] change `skip` parameter to `from` - `till` (or some variation of wording)
-   [ ] investigate use `JAX` or `QNumeric` as alternatives to `Numba`
-   [ ] investigate implementation of `CUDA` kernels for GPU acceleration using `Numba`

## [WIP] Citation

Add Zenodo link as soon as a first public release is planned to coincide with open-sourcing.

## Contributors

-   Emile de Bruyn

## [WIP] License

### To-dos

-   add LGPL license
-   check with colleagues and legal department before publication
