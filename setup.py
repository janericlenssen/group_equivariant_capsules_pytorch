from setuptools import setup, find_packages

__version__ = '0.0.0'

setup(
    name='group-capsules',
    version=__version__,
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    packages=find_packages())
