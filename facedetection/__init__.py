from setuptools_scm import get_version
from logging import (
    root, INFO, StreamHandler,
)

try:
    __version__ = get_version()

except:
    __version__ = "0.0.0"

    logger = root
    logger.setLevel(INFO)
    logger.addHandler(StreamHandler())

    logger.warning("Not able to get version.")