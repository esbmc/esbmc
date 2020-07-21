# The ESBMC model checker

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/d14d06e975644907a2eb9521e09ccfe4)](https://app.codacy.com/gh/esbmc/esbmc?utm_source=github.com&utm_medium=referral&utm_content=esbmc/esbmc&utm_campaign=Badge_Grade_Dashboard)
[![Build Status](https://travis-ci.com/esbmc/esbmc.svg?branch=master)](https://travis-ci.com/esbmc/esbmc)
![Lint Code Base](https://github.com/esbmc/esbmc/workflows/Lint%20Code%20Base/badge.svg)
![Health Checks](https://github.com/esbmc/esbmc/workflows/Health%20Checks/badge.svg)
![Build All Solvers](https://github.com/esbmc/esbmc/workflows/Build%20All%20Solvers/badge.svg)
[![codecov](https://codecov.io/gh/esbmc/esbmc/branch/master/graph/badge.svg)](https://codecov.io/gh/esbmc/esbmc)




ESBMC, the efficient SMT based model checker, is a software verification tool for C and C++ code bases. The technique is sound but incomplete -- an error found by ESBMC will be correct (modulo errors in the tool), but a successful verification does not guarantee there are no errors.

To build ESBMC, please see the BUILDING file. For getting started, we recommend first reading some of the background material / publications, to understand exactly what this technique can provide, for example our SV-COMP tool papers.

The cannonical public location of ESBMCs source is on github:

    https://github.com/esbmc/esbmc

While our main website is esbmc.org

### Features

ESBMC aims to support all of C99, and detects errors in software by simulating a finite prefix of the program execution with all possible inputs. Classes of problems that can be detected include:
 * User specified assertion failures
 * Out of bounds array access
 * Illegal pointer dereferences, such as:
   * Dereferencing null
   * Performing an out-of bounds dereference
   * Double-free of malloc'd memory
   * Misaligned memory access
 * Integer overflows
 * Divide by zero
 * Memory leaks

Concurrent software (using the pthread api) is verified by explicit exploration of interleavings, producing one symbolic execution per interleaving. By default only normal errors will be checked for; one can also specify options to check concurrent programs for:
 * Deadlock (only on pthread mutexes and convars)
 * Data races (i.e. competing writes)

By default ESBMC performs a "lazy" depth first search of interleavings -- it can also encode (explicitly) all interleavings into a single SMT formula.

A number of SMT solvers are currently supported:
 * Z3 4.8+
 * Boolector 3.0+
 * MathSAT
 * CVC4
 * Yices 2.2+

In addition, ESBMC can be configured to use the SMTLIB interactive text format with a pipe, to communicate with an arbitary solver process, although not-insignificant overheads are involved.

A limited subset of C++98 is supported too -- a library modelling the STL is also available.

### Differences from CBMC

ESBMC is based on CBMC, the C bounded model checker. The primary differences between the two are that CBMC focuses on SAT-based encodings of unrolled C programs while ESBMC targets SMT; and CBMC's concurrency support is a fully symbolic encoding of a concurrent program in one SAT formulae.

The fundemental verification technique (unrolling programs to SSA then converting to a formula) is still the same in ESBMC, although the program internal representation has been had some additional types added. ESBMC also implements a state-of-the-art k-induction proof rule to falsify and prove safety properties in C/C++ programs.

# Open source

ESBMC has now been released as open source software -- mainly distributed under the terms of the Apache License 2.0. ESBMC contains a signficant amount of other peoples software, however, please see the COPYING file for an explanation of who-owns-what, and under what terms they are distributed.

We'd be extremely happy to receive contributions to make ESBMC better (under the terms of the Apache License 2.0). If you'd like to submit anything, please file a pull request against the public github repo. General discussion and release announcements will be made via GitHub. To contact us about research or collaboration, please post an issue in GitHub.

### Getting started

Currently, we don't have a good guide for getting started with ESBMC, although we hope to improve this in the future. Examining some of the benchmarks in the SV-COMP competition (http://sv-comp.sosy-lab.org/) would be a good start, using the esbmc command line for the relevant competition year.

### Contributing to the code base

Here are some steps to contributing to the code base:

  1. Compile and execute esbmc. [Building](https://github.com/esbmc/esbmc/blob/master/BUILDING.md)
  1. Fork the repository
  1. Clone the repository git clone git@github.com:YOURNAME/esbmc.git
  1. Create a branch from the master branch (default branch)
  1. Make your changes
  1. Check the formatting with clang-format (use Clang 9)
  1. Push your changes to your branch
  1. Create a Pull Request targeting the master branch

Here is an example to prepare a pull request (PR) 


A) Make sure that you are in the `master` branch and your fork is updated.

```
git checkout master
git fetch upstream
git pull --rebase upstream master    
git push origin HEAD:master
```

Note that if you have not yet setup the `upstream`, you need to type the following command:

```
git remote add upstream https://github.com/esbmc/esbmc
```

B) Create a local branch (e.g., `model-pthread-create`) from the `master` branch:

```
git checkout -b model-pthread-equal --track master
```

C) Add your changes via commits to the local branch:

```
git add path-to-file/file.cpp
git commit -sm "added opertational model for pthread_equal"
```

Note that you can check your changes via `git status`. 
Note also that every PR should contain at least two test cases 
to check your implementation: one successful and one failed test case.

D) Push your changes in the local branch to the ESBMC repository:

```
git push origin model-pthread-equal
```

New contributors can check issues marked with `good first issue` by clicking [here](https://github.com/esbmc/esbmc/contribute).

### Documentation

A limited number of classes have been marked up with doxygen documentation headers. Comments are put in the header files declaring classes and methods. HTML documation can be generated by running:

    doxygen .doxygen

Output will be in docs/html, open index.html to get started. 
