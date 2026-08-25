from pathlib import Path

from setuptools import find_packages, setup


# --------------------------------------------------
# README
# --------------------------------------------------

this_directory = Path(__file__).parent

long_description = (
    this_directory / "README.md"
).read_text(
    encoding="utf-8"
)


# --------------------------------------------------
# PACKAGE
# --------------------------------------------------

setup(
    name="IntraSOM",

    version="1.1.2",

    author=(
        "InTRA RDI Center "
        "(Universidade de São Paulo)"
    ),

    author_email="intra@usp.br",

    description=(
        "IntraSOM: Library for Self-Organizing Maps "
        "with missing data, rectangular and hexagonal "
        "lattices, and planar or toroidal topology"
    ),

    long_description=long_description,

    long_description_content_type="text/markdown",

    keywords=(
        "IntraSOM SOM Self-Organizing Maps "
        "Kohonen Unsupervised Learning "
        "U-Matrix Toroidal Hexagonal Rectangular"
    ),

    packages=find_packages(),

    python_requires=">=3.11",

    install_requires=[

        # Core numerical stack
        "numpy>=1.26.4",
        "scipy>=1.11.4",
        "pandas>=2.1.4",

        # Machine learning
        "scikit-learn>=1.4.0",
        "joblib>=1.2.0",

        # Visualization
        "matplotlib>=3.8.2",
        "plotly>=5.14.1",
        "Pillow>=10.0.0",
        "scikit-image>=0.22.0",

        # Progress
        "tqdm>=4.65.0",

        # Data formats
        "pyarrow>=14.0.2",
        "openpyxl>=3.1.2",

        # Notebook support
        "ipywidgets>=8.0.6",
        "nbformat>=5.9.0",

        # Geospatial
        "shapely>=2.0.2",
        "geopandas>=0.14.1",
    ],

    package_data={
        "intrasom": [
            "images/*.jpg",
            "images/*.svg",
            "images/*.png",
        ]
    },

    include_package_data=True,
)