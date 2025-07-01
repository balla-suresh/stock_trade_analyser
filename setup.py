#!/usr/bin/env python3
import os
import pathlib
import shutil
import setuptools
from setuptools import Command

NAME = "trade"
DESC = "Trade utilities"

with open("requirements/base.txt") as f:
    REQUIREMENTS = f.read().splitlines()


python_requires = "==3.12"

setuptools.setup(
    name=NAME,
    author="suresh",
    author_email="sballa@vmware.com",
    version="1.0",
    description=DESC,
    install_requires=REQUIREMENTS,
    packages=setuptools.find_packages('lib')
)
