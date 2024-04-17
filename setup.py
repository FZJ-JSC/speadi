from setuptools import setup, find_packages

__version__ = '1.0.0'

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
    long_description=README,
    url="https://github.com/FZJ-JSC/speadi",
)
