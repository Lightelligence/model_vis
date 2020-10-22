"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""
import os
import pathlib

# Always prefer setuptools over distutils
from setuptools import find_packages, setup

req_path = "requirements.txt"
if os.path.exists(req_path):
    with open(req_path) as f:
        requirements = [r.strip() for r in f.readlines()]
else:
    requirements = []

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")
# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(name="model_vis",
      version="0.0.22",
      python_requires=">3.0.0",
      description=
      "An easy and interactive graph visualization tool for ML models!!!",
      long_description=long_description,
      long_description_content_type="text/markdown",
      author_email="amipro@gmail.com",
      install_requires=requirements,
      keywords="ML, visualization, visualize, model, graph",
      packages=find_packages(),
      include_package_data=True,
      entry_points={
          "console_scripts": [
              "plot=graph_visualization.plot_graph:main",
          ],
      })
