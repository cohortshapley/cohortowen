from setuptools import setup, find_packages
import sys

setup(name='cohortshapley',
      packages=[package for package in find_packages()
                if package.startswith('cohortowen')],
      install_requires=[
          'numpy',
          'pandas',
          'itertools',
          'tqdm_pathos',
          'math',
          'sklearn',
          'urllib',
          'os'
      ],
      
      description='Cohort Owen',
      author='Benjamin Seiler',
      url='https://github.com/cohortshapley/cohortowen',
      author_email='bbseiler@stanford.edu',
      license='MIT',
      version='0.1.0')
