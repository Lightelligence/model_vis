"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

import pathlib

# Always prefer setuptools over distutils
from setuptools import find_packages, setup

# here = pathlib.Path(__file__).parent.resolve()
# print("PATH TO HERE:", here)
# # Get the long description from the README file
# long_description = (here / 'README.md').read_text(encoding='utf-8')
# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name='model_vis',
    version='0.0.1',
    description=
    'An easy and interactive graph visualization tool for ML models!!!',
    author_email='amipro@gmail.com',
    keywords='ML, visualization, visualize, model, graph',
    packages=find_packages(),
    # packages=['graph_visualization'],
    entry_points={
        'console_scripts': [
            'plot=graph_visualization.plot_graph:main',
        ],
    })
