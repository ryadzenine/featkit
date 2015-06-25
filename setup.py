#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

import featkit

setup(
    name='featkit',
    version=featkit.__version__,
    packages=find_packages(),
    author="Ryad Zenine",
    author_email="r.zenine@gmail.com",
    description="Feature engineering library compatible with scikit-learn",
    long_description=open('README.md').read(),
    install_requires=["scikit-learn"],
    include_package_data=True,
    url='http://github.com/ryadzenine/featkit',
    classifiers=[
        "Programming Language :: Python",
        "Development Status :: 1 - Alpha",
        "License :: OSI Approved",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 2.7",
        "Topic :: Scientific/Engineering",
    ],
    license="BSD",
)
