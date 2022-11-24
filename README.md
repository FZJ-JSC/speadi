<h1 class="title"> SPEADI <br /> </h1>

A Python package that aims to characterise the dynamics of local chemical environments from Molecular Dynamics trajectories of proteins and other biomolecules.

<a href="https://gitlab.jsc.fz-juelich.de/debruyn1/speadi/-/commits/master"><img alt="pipeline status" src="https://gitlab.jsc.fz-juelich.de/debruyn1/speadi/badges/master/pipeline.svg" /></a>  <a href="https://gitlab.jsc.fz-juelich.de/debruyn1/speadi/-/commits/master"><img alt="coverage report" src="https://gitlab.jsc.fz-juelich.de/debruyn1/speadi/badges/master/coverage.svg" /></a>

# Introduction

speadi provides the user tools with which to characterise the local chemical environment using Molecular Dynamics data. At the moment, it implements two variations of the pair radial distribution function (RDF): time-resolved RDFs (TRRDFs) and van Hove dynamic correlation functions (VHFs).

# Quick install

For installation into the default python environment, run the following pip command in a terminal:

```bash
pip install git+https://github.com/EmileDeBruyn/speadi.git
```

Or, to install just into the current user's local environment, add the `--user` option:

```bash
pip install --user git+https://github.com/EmileDeBruyn/speadi.git
```

# RDFs

Normally, Radial Distribution Functions used in atomistic simulations are averaged over entire trajectories. `speadi` averages over user-defined windows of time. This gives a separate RDF between group *a* and *b* for each window in the trajectory.

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

## Acceleration

As a default, speadi doesn't install `JAX` or `Numba`, but uses these if detected in the same Python environment that `speadi` is installed into.

To install `JAX` and `jaxlib` along with `speadi`, simply add the `jax` extra to `pip`:

```bash
pip install 'git+https://github.com/EmileDeBruyn/speadi.git#egg=SPEADI[jax]'
```

Note that by default, installing `jax` using pip (through pypi) only enables CPU acceleration. To enable GPU or TPU acceleration, please see <https://github.com/google/jax> for details on how to obtain a `JAX` installation for the specific `CuDNN` version in your environment.

To install `Numba` along with `speadi`, simply add the `numba` extra to `pip`:

```bash
pip install 'git+https://github.com/EmileDeBruyn/speadi.git#egg=SPEADI[numba]'
```

Or, to install both `jax` and `numba` alongside `speadi`, add the `all` extra to `pip`:

```bash
pip install 'git+https://github.com/EmileDeBruyn/speadi.git#egg=SPEADI[all]'
```

The `--user` pip option may be added to all of these commands to install just for the current user.

## Editable source installation

Open up a terminal. Navigate to the location you want to clone this repository. Then, run the following to clone the entire repository:

```bash
git clone https://github.com/EmileDeBruyn/speadi
```

Then, install locally using `pip` by adding the `-e` option:

```bash
pip install -e speadi
```

## Usage

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

Now you can load `speadi` to obtain RDFs:

```python
import speadi as sp
```

To make an RDF for each heavy protein atom

```python
r, g_rt = mde.trrdf(trajectory, protein_by_atom, [na, cl], top=top, n_windows=1000, window_size=500,\
              skip=0, pbc='general', stride=1, nbins=400)
```

To repeat the analysis, but obtain integral of $g(r)$ instead, simply replace `trrdf` with `int_trrdf` instead.

```python
r, n_rt = mde.int_trrdf(trajectory, protein_by_atom, [na, cl], top=top, n_windows=1000, window_size=500,\
              skip=0, pbc='general', stride=1, nbins=400)
```

## [WIP] Citation

Add Zenodo link as soon as a first public release is planned to coincide with open-sourcing.

## Acknowledgments

We gratefully acknowledge the following institutions for their support in the development of speadi and for granting compute time to develop and test speadi.

-   Gauss Centre for Supercomputing e.V. (www.gauss-centre.eu) and the John von Neumann Institute for Computing (NIC)

on the GCS Supercomputer JUWELS at Jülich Supercomputing Centre (JSC)

-   HDS-LEE Helmholtz Graduate School

## Contributors

-   Emile de Bruyn

## Copyright

speadi Copyright (C) 2022 Forschungszentrum Jülich GmbH, Jülich Supercomputing Centre and the Authors

## License

This library is free software; you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation; either version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
