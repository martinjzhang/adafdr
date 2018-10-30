from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install
from subprocess import check_call
from distutils.command.build_py import build_py as _build_py
import sys, os.path

setup(name='adafdr',
      version='0.1.0',
      description='A fast and covariate-adaptive method for multiple hypothesis testing',
      url='https://github.com/martinjzhang/adafdr',
      author='Martin Zhang, Fei Xia, James Zou',
      author_email='jinye@stanford.edu',
      license='Stanford University',
      packages=['adafdr'],
      zip_safe=False)