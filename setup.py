from setuptools import setup, find_packages
import sys

setup(name='cohortowen',
      packages=[package for package in find_packages()
                if package.startswith('cohortowen')],
      install_requires=[
          'numpy',
          'pandas',
          'tqdm_pathos',
          'scikit-learn',
      ],
      
      description='Cohort Owen',
      author='Benjamin Seiler',
      url='https://github.com/cohortshapley/cohortowen',
      author_email='bbseiler@stanford.edu',
      license='MIT',
      version='0.1.0')
