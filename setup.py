try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="IntraSOM",
    version="1.0",
    author="Rodrigo Gouvêa",
    description="Self Organizing Maps Package with NaN input values and Position Matrix",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "matplotlib==3.7.1",
        "scipy==1.10.1",
        "joblib==1.2.0",
        "scikit-learn==1.2.2",
        "pandas==2.0.1",
        "tqdm==4.65.0",
        "plotly==5.14.1",
        "scikit-image==0.20.0",
        "pyarrow>=9.0.0",
        "openpyxl==3.1.2"
    ],
    package_data={
        'intrasom': ['images/*.jpg']  # include all jpg files under the Images directory
    }
)
