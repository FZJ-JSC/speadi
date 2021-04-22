- [Time-resolved RDF.py](#sec-1)
  - [Install](#sec-1-1)
    - [Direct installation](#sec-1-1-1)
    - [Editable source installation](#sec-1-1-2)


# Time-resolved RDF.py<a id="sec-1"></a>

## Install<a id="sec-1-1"></a>

Installation is provided easily through `pip`. It can be installed either directly as a package, or as an editable source.

### Direct installation<a id="sec-1-1-1"></a>

For installation into the default python environment, run the following in a terminal:

```bash
pip install git+https://gitlab.version.fz-juelich.de/debruyn1/emiles-phd-project.git#egg=version_subpkg\&subdirectory=scripts/vanhove
```

To install for just the current user, add the `--user` option:

```bash
pip install --user git+https://gitlab.version.fz-juelich.de/debruyn1/emiles-phd-project.git#egg=version_subpkg\&subdirectory=scripts/vanhove
```

### Editable source installation<a id="sec-1-1-2"></a>

Open up a terminal. Navigate to the location you want to clone this repository. Then, run the following to clone the whole repository:

```bash
git clone ssh://git@gitlab.version.fz-juelich.de:10022/debruyn1/emiles-phd-project.git
```

Then, install locally using `pip` (after entering the package sub-directory):

```bash
cd emiles-phd-project/scripts/time_resolved_rdf/ && pip install -e .
```
