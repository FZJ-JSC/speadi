from setuptools import setup, find_packages
from pathlib import Path


__version__ = '1.0.2'

this_directory = Path(__file__).parent
long_description = (this_directory / 'README.md').read_text()

setup(
    name='SPEADI',
    version=__version__,
    packages=find_packages(),
    author='Emile de Bruyn',
    author_email='e.de.bruyn@fz-juelich.de',
    description="""
    A Python package that aims to characterise the dynamics of local chemical environments
    from Molecular Dynamics trajectories of proteins and other biomolecules.
    """,
    long_description_content_type="text/markdown",
    long_description=long_description,
    url="https://github.com/FZJ-JSC/speadi",
)
