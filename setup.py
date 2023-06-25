try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="IntraSOM",
    version="1.0",
    author="InTRA RDI Center (Universidade de São Paulo)",
    author_email="intra@usp.br",
    description="IntraSOM Library for Self-Organizing Maps with missing data, hexagonal lattice and toroidal projection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="Intrasom SOM Self-Organization Kohonen Non-Supervised U-Matrix",
    packages=find_packages(),
    install_requires=[
        "matplotlib>=3.7.1",
        "scipy>=1.10.1",
        "joblib>=1.2.0",
        "scikit-learn>=1.2.2",
        "pandas>=2.0.1",
        "tqdm>=4.65.0",
        "plotly>=5.14.1",
        "scikit-image>=0.20.0",
        "pyarrow>=9.0.0",
        "openpyxl>=3.1.2",
        "pyarrow >= 9.0.0",
        "openpyxl >= 3.1.2",
        "ipywidgets >= 8.0.6",
        "shapely >= 2.0.1",
        "geopandas >= 0.13.0"
    ],
    package_data={
        'intrasom': ['images/*.jpg', 'images/*.svg', 'images/*.png']
    }
)
