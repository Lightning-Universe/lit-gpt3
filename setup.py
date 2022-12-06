#!/usr/bin/env python

from setuptools import find_packages, setup

PACKAGE_NAME = "lightning_gpt3"

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name=PACKAGE_NAME,
    version="0.0.1",
    description="",
    author="",
    author_email="",
    url="",
    packages=find_packages(exclude=["tests", "docs"]),
    long_description="",
    long_description_content_type="text/markdown",
    include_package_data=True,
    zip_safe=False,
    keywords=["deep learning", "pytorch", "AI"],
    python_requires=">=3.8",
    setup_requires=["wheel"],
    install_requires=requirements,
)
