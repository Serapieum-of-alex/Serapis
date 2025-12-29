![GitHub release (latest by date)](https://img.shields.io/github/v/release/Serapieum-of-alex/serapis)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5758979.svg)](https://doi.org/10.5281/zenodo.5758979)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Serapieum-of-alex/serapis/master)
[![Python Versions](https://img.shields.io/pypi/pyversions/serapis.png)](https://img.shields.io/pypi/pyversions/serapis)
[![Documentation Status](https://readthedocs.org/projects/serapis/badge/?version=latest)](https://serapis.readthedocs.io/en/latest/?badge=latest)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/Serapieum-of-alex/serapis.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/Serapieum-of-alex/serapis/context:python)


[![GitHub Clones](https://img.shields.io/badge/dynamic/json?color=success&label=Clone&query=count&url=https://github.com/Serapieum-of-alex/serapis/blob/master/clone.json?raw=True&logo=github)](https://github.com/MShawon/github-clone-count-badge) [![Say Thanks!](https://img.shields.io/badge/Say%20Thanks-!-1EAEDB.svg)](https://saythanks.io/to/Serapieum-of-alex)

Current build status
====================


<table><tr><td>All platforms:</td>
    <td>
      <a href="https://dev.azure.com/conda-forge/feedstock-builds/_build/latest?definitionId=12419&branchName=master">
        <img src="https://dev.azure.com/conda-forge/feedstock-builds/_apis/build/status/serapis-feedstock?branchName=master">
      </a>
    </td>
  </tr>
</table>

[![Build status](https://ci.appveyor.com/api/projects/status/rys2u0l1nbmfjuww?svg=true)](https://ci.appveyor.com/project/Serapieum-of-alex/serapis)
[![codecov](https://codecov.io/gh/Serapieum-of-alex/serapis/branch/main/graph/badge.svg?token=EMQSR7K2YV)](https://codecov.io/gh/Serapieum-of-alex/serapis)
![GitHub last commit](https://img.shields.io/github/last-commit/Serapieum-of-alex/serapis)
![GitHub forks](https://img.shields.io/github/forks/Serapieum-of-alex/serapis?style=social)
![GitHub Repo stars](https://img.shields.io/github/stars/Serapieum-of-alex/serapis?style=social)

[![Github all releases](https://img.shields.io/github/downloads/Naereen/StrapDown.js/total.svg)](https://GitHub.com/Naereen/StrapDown.js/releases/)


Current release info
====================

| Name | Downloads                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Version | Platforms |
| --- |-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| --- | --- |
| [![Conda Recipe](https://img.shields.io/badge/recipe-serapis-green.svg)](https://anaconda.org/conda-forge/serapis) | [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/serapis.svg)](https://anaconda.org/conda-forge/serapis) [![Downloads](https://pepy.tech/badge/serapis)](https://pepy.tech/project/serapis) [![Downloads](https://pepy.tech/badge/serapis/month)](https://pepy.tech/project/serapis)  [![Downloads](https://pepy.tech/badge/serapis/week)](https://pepy.tech/project/serapis)  ![PyPI - Downloads](https://img.shields.io/pypi/dd/serapis?color=blue&style=flat-square) ![GitHub all releases](https://img.shields.io/github/downloads/Serapieum-of-alex/serapis/total) | [![Conda Version](https://img.shields.io/conda/vn/conda-forge/serapis.svg)](https://anaconda.org/conda-forge/serapis) [![PyPI version](https://badge.fury.io/py/serapis.svg)](https://badge.fury.io/py/serapis) [![Anaconda-Server Badge](https://anaconda.org/conda-forge/serapis/badges/version.svg)](https://anaconda.org/conda-forge/serapis) | [![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/serapis.svg)](https://anaconda.org/conda-forge/serapis) [![Join the chat at https://gitter.im/serapis/serapis](https://badges.gitter.im/serapis/serapis.svg)](https://gitter.im/serapis/serapis?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) |


![serapis](/docs/img/serapis4.png) ![serapis](/docs/img/name.png)


serapis - Hydrological library for Python
=====================================================================
**serapis** is an open-source Python Framework for building raster-based conceptual distributed hydrological models using HBV96 lumped
model & Muskingum routing method at a catchment scale (Farrag & Corzo, 2021), serapis gives a high degree of flexibility to all components of the model
(spatial discretization - cell size, temporal resolution, parameterization approaches and calibration (Farrag et al., 2021)).


![1](/docs/img/Picture1.png)  ![2](/docs/img/Picture2.png)

serapis


Installing serapis
===============

Installing `serapis` from the `conda-forge` channel can be achieved by:

```
conda install -c conda-forge serapis
```

It is possible to list all of the versions of `serapis` available on your platform with:

```
conda search serapis --channel conda-forge
```

## Install from Github
to install the last development to time you can install the library from github
```
pip install git+https://github.com/Serapieum-of-alex/serapis
```

## pip
to install the last release you can easly use pip
```
pip install serapis==0.1.0
```

Quick start
===========

```
  >>> import serapis
```

[other code samples](https://serapis.readthedocs.io/en/latest/?badge=latest)

## Naming Convention
[PEP8](https://peps.python.org/pep-0008/#naming-conventions)
- module names: lower case word, preferably one word if not, separate words with underscores (module.py, my_module.py).
- class names: PascalCase (Model, MyClass).
- class method/function: CamelCase(getFile, readConfig).should have a verb one them, because they perform some action
