---
title: Linear Temporal Logic
---

The algorithms to support LTL in ESBMC have been described in [Jeremy Morse's thesis](https://ssvlab.github.io/esbmc/papers/phd_thesis_morse.pdf). He also contributed the initial implementations, both in ESBMC and in the support library ltl2c. This library is based on [ltl2ba](http://www.lsv.fr/~gastin/ltl2ba/) and was used to transform a user-defined LTL formula to a C implementation of a Büchi automaton, checking the validity of (the negation of) the formula on finite prefixes of a certain kind of omega words. See Jeremy's thesis for details.

The ltl2c library has since been merged with the latest developments of ltl2ba in the [libltl2ba](https://github.com/esbmc/libltl2ba) project, among various fixes and extensions. See its [Changelog](https://github.com/esbmc/libltl2ba/blob/master/CHANGELOG) for details.

## LTL checking with ESBMC

Given a C program `program.c` with externally visible state variables and/or predicates $\vec\tau=(\tau_1,\ldots,\tau_n)$ declared in a header file `tau.h`, and an LTL formula $\varphi$ over $\vec\tau$, the procedure of checking the validity of $\varphi$ on program traces of `program.c` is as follows.

1. Run `ltl2ba -O c -f '!(phi)' > notphi.c` where `phi` is the textual representation of $\varphi$ in libltl2ba's grammar (see [README](https://github.com/esbmc/libltl2ba/blob/master/README) for details on the grammar). This results in a new file `notphi.c` encoding the Büchi automaton accepting $\neg\varphi$ in a C dialect using some of ESBMC's intrinsics.
2. Run `esbmc --ltl program.c notphi.c --include-file tau.h`. This results in one of four outcomes, reported by ESBMC in a log message starting with `Final lowest outcome:`
   - LTL_BAD or $\bot$: a trace of `program.c` has been found that violates $\varphi$.
   - LTL_FAILING or $\bot^p$: a trace of `program.c` ends in a state that would violate $\varphi$ when stutter extended.
   - LTL_SUCCEEDING or $\top^p$: `program.c` halts in a state that satisfies $\varphi$ when stutter extended.
   - LTL_GOOD or $\top$: `program.c` satisfies $\varphi$.

See Jeremy's thesis, section 3.2.5, for details on $\top$, $\top^p$ and the respective $\bot$ variants; and section 3.1.1.1 for details on the stutter extension of a finite trace.

## Status of LTL in ESBMC

The LTL checking code, which had been removed from ESBMC earlier, has been restored. Previously, the external variables in $\vec\tau$ would be stored as strings in a special variable in `notphi.c`, and ESBMC would parse these strings as C expressions. This capability has been removed because the frontends no longer support mapping strings to ESBMC expressions directly. To work around this, libltl2ba and ESBMC's LTL support have been extended to generate and use pure C functions with particular identifiers in `notphi.c`, computing the results of evaluating particular C expressions.

The LTL support in ESBMC has not yet been extensively tested since it was reintroduced.

## Example
`ltlba -O c -f '{state}'` generates the following pure C function for accessing the extern C variable `state`:
```c
int _ltl2ba_cexpr_0_status(void) { return state; }
```
In order to run ESBMC on the resulting C program, checking the Büchi automaton's acceptance condition, a header file
```c
extern int state;
```
needs to be provided in addition to the program modifying `state` itself.
