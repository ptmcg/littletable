#!/usr/bin/env python

"""Setup script for the littletable module distribution."""
from setuptools import setup

from littletable import __version__ as littletable_version

with open('README.md') as readme:
    long_description_text = readme.read()

modules = ["littletable"]

setup(# Distribution meta-data
    name="littletable",
    version=littletable_version,
    description="Python in-memory ORM database",
    long_description=long_description_text,
    long_description_content_type='text/markdown',
    author="Paul McGuire",
    author_email="ptmcg@austin.rr.com",
    license="MIT License",
    url="https://github.com/ptmcg/littletable/",
    download_url="https://pypi.org/project/littletable/",
    py_modules=modules,
    python_requires=">=3.9",
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Topic :: Database',
        ]
    )
