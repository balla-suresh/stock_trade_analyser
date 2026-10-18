#!/usr/bin/env python3
import setuptools

NAME = "trade"
DESC = "Trade utilities"

with open("requirements/base.txt") as f:
    REQUIREMENTS = f.read().splitlines()


setuptools.setup(
    name=NAME,
    author="suresh",
    author_email="sballa@vmware.com",
    version="1.0",
    description=DESC,
    install_requires=REQUIREMENTS,
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    package_data={
        "stock_trade_analyser": ["config/*.json"],
    },
    python_requires=">=3.12",
)
