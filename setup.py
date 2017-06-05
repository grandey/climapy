"""
climapy setup.py:
    Setup file for climapy.

Installation of climapy:
    python setup.py install
    
Author:
    Benjamin S. Grandey, 2017
"""

from os import path
from setuptools import setup
from subprocess import Popen, PIPE

here = path.abspath(path.dirname(__file__))

# Description and long description
description = 'Support data analysis of climate model data.'
with open(path.join(here, 'README.md')) as f:
    long_description = f.read()

# Version - add info about git revision
version_prefix = '0.0.1'
with Popen(['git', 'describe', '--always'], stdout=PIPE) as p:
    git_revision = p.stdout.read().strip().decode('utf-8')
with Popen(['git', 'status', '--porcelain'], stdout=PIPE) as p:
    git_changes = p.stdout.read().strip().decode('utf-8')
    if git_changes:
        git_revision = '{}.uncommitted_changes'.format(git_revision)
version = '{}+{}'.format(version_prefix, git_revision)

# Create of version.py
version_filename = path.join(here, 'climapy/version.py')
with open(version_filename, 'w') as f:
    f.write('"""climapy version.py: automatically created by setup.py"""\n')
    f.write('\n')
    f.write("__all__ = ['__version__', ]\n")
    f.write('\n')
    f.write("__version__ = '{}'\n".format(version))

# Setup
setup(
    name='climapy',
    description=description,
    long_description=long_description,
    version=version,
    author='Benjamin S. Grandey',
    author_email='benjamin@smart.mit.edu',
    url='https://github.com/grandey/climapy',
    license='MIT',
    classifiers=['Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.6'],
    packages=['climapy', ],
    install_requires=['numpy', 'xarray', 'scipy']
)
