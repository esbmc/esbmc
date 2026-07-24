---
title: Setup
weight: 1
---

## Homebrew (macOS and Linux)

On macOS or Linux with [Homebrew](https://brew.sh), install ESBMC with:

```sh
brew install esbmc
```

This pulls in the bundled SMT solvers (Z3, Bitwuzla) and puts `esbmc` on your
`PATH`.

## Prebuilt binaries

Alternatively, download the latest binary for Linux, Windows or macOS from
[GitHub](https://github.com/esbmc/esbmc/releases), then save and unzip it on your
disk.

Once unzipped, read the license before running ESBMC. The distribution is split
into two directories:

- `bin`: the static ESBMC binary;
- `license`: the ESBMC, Z3 and Boolector licenses.

If you want to use other SMT solvers (e.g. MathSAT, Yices, CVC4), check out the
ESBMC [source code](https://github.com/esbmc/esbmc) and follow the
[Build Guide](/docs/development/building).
