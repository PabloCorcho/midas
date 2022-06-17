from setuptools import setup, find_packages

setup(
   name='midas',
   version='0.1.0',
   author='Pablo Corcho-Caballero',
   author_email='pablo.corcho@uam.es',
   #packages=['package_name', 'package_name.test'],
   #scripts=['bin/script1','bin/script2'],
   url='http://pypi.python.org/pypi/PackageName/',
   license='LICENSE.txt',
   description='Create mock IFS observations using hydro sims!',
   long_description=open('README.md').read(),
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: GNU License",
    "Operating System :: OS Independent",
  ],
  package_dir={"": "src"},
  include_package_data=True,
  python_requires=">=3.6",
)