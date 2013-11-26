#coding: utf8

"""
Setup script for pypad.
"""

from glob import glob


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(name='cxsinfer',
      version='0.0.1',
      author="TJ Lane",
      author_email="tjlane@stanford.edu",
      description='Inference of electron density from CXS using sparse sensing',
      packages=["cxsinfer"],
      package_dir={"cxsinfer": "cxsinfer"},
      #scripts=[s for s in glob('scripts/*') if not s.endswith('__.py')],
      test_suite="test")
