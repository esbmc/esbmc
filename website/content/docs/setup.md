---
title: Setup
weight: 1
---

## Ubuntu

The easiest way to install ESBMC on Ubuntu is through our official
[PPA](https://launchpad.net/~esbmc/+archive/ubuntu/esbmc), which provides
releases for automatic installation:

```sh
sudo add-apt-repository ppa:esbmc/esbmc
sudo apt update
sudo apt install esbmc
```

This method is recommended for general users and supports Ubuntu 22.04 (Jammy)
and 24.04 (Noble).

## Homebrew (macOS and Linux)

```sh
brew install esbmc
```

This installs `esbmc` together with its bundled SMT solvers (Z3, Bitwuzla).

## GitHub Release

You can also download the latest binary for Linux, Windows or macOS from the
[releases page](https://github.com/esbmc/esbmc/releases), then save and unzip it
on your disk.

Once unzipped, read the license before running ESBMC. The distribution is split
into two directories:

- `bin`: the static ESBMC binary;
- `license`: the ESBMC, Z3 and Boolector licenses.

If you want to use other SMT solvers (e.g. MathSAT, Yices, CVC4), check out the
ESBMC [source code](https://github.com/esbmc/esbmc) and follow the
[Build Guide](/docs/development/building).
