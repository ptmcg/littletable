#!/usr/bin/env python

"""Setup script for the littletable module distribution."""
from distutils.core import setup

import sys
import os
from littletable import __version__ as littletable_version

_PY3 = sys.version_info[0] > 2
    
modules = ["littletable",]

setup(# Distribution meta-data
    name = "littletable",
    version = littletable_version,
    description = "Python in-memory ORM database",
    author = "Paul McGuire",
    author_email = "ptmcg@users.sourceforge.net",
    license = "MIT License",
    url = "http://littletable.sourceforge.net/",
    py_modules = modules,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        ]
    )
