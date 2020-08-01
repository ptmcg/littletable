
#!/usr/bin/env python

"""Setup script for the littletable module distribution."""
from setuptools import setup

import sys
import os
from littletable import __version__ as littletable_version

_PY3 = sys.version_info[0] > 2

with open('README.md') as readme:
    long_description_text = readme.read()

modules = ["littletable",]

setup(# Distribution meta-data
    name = "littletable",
    version = littletable_version,
    description = "Python in-memory ORM database",
    long_description = long_description_text,
    long_description_content_type = 'text/markdown',
    author = "Paul McGuire",
    author_email = "ptmcg@austin.rr.com",
    license = "MIT License",
    url = "https://github.com/ptmcg/littletable/",
    download_url = "https://pypi.org/project/littletable/",
    py_modules = modules,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        ]
    )
