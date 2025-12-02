""" A helping lib for writing descent simulators """

import importlib.metadata

# retrieves version from pyproject.toml once installed using pip
try:
    __version__ = importlib.metadata.version('descentimagegenerator')
except importlib.metadata.PackageNotFoundError:
    __version__ = "'descentimagegenerator' not installed with pip, __version__ will only be available once it is installed"


#Declaring main API
from .trajectoryrenderer import TrajectoryRenderer

__all__ = ["TrajectoryRenderer"]