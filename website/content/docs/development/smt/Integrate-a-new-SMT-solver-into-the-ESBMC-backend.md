---
title: Implementing SMT Solver
---


In order to integrate a new SMT solver into the ESBMC backend, we would need to follow these steps:

1) Copy and paste this directory:

https://github.com/esbmc/esbmc/tree/master/src/solvers/z3

2) Replace everything with "z3" by "NewSolver".

3) Replace the calls to the Z3 SMT solver methods with those from your `NewSolver` in https://github.com/esbmc/esbmc/tree/master/src/solvers/z3.

4) Update https://github.com/esbmc/esbmc/blob/master/src/solvers/z3/CMakeLists.txt.

5) Update
https://github.com/esbmc/esbmc/blob/master/src/solvers/CMakeLists.txt#L24
https://github.com/esbmc/esbmc/blob/master/src/solvers/solve.cpp#L10
https://github.com/esbmc/esbmc/blob/master/src/solvers/solve.cpp#L40

6) Update https://github.com/esbmc/esbmc/blob/master/BUILDING.md

7) Update:
https://github.com/esbmc/esbmc/blob/7f41ac14baf26ca32521781a9404105413dc61a4/src/esbmc/options.cpp#L93-L97

8) Update
https://github.com/esbmc/esbmc/blob/7f41ac14baf26ca32521781a9404105413dc61a4/src/esbmc/esbmc_parseoptions.cpp#L1752

9. Update:
https://github.com/esbmc/esbmc/blob/7f41ac14baf26ca32521781a9404105413dc61a4/src/solvers/solver_config.h.in#L1

10. Integrate the solver into the CI: [Build](https://github.com/esbmc/esbmc/blob/master/.github/workflows/build.yml) and [Release](https://github.com/esbmc/esbmc/blob/master/.github/workflows/release.yml)