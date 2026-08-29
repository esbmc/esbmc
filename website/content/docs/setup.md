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

### ARM64 builds

`esbmc-linux-armv8.zip` is the Linux ARM64 (aarch64) build. It is static and
self-contained like the x86_64 build, but a few features are unavailable on
ARM64. Each is tracked by a GitHub issue; if the issue is closed, the limitation
has been lifted and this page is simply out of date.

| Limitation                                    | Tracked by |
| --------------------------------------------- | ---------- |
| `--32` fails: no 32-bit libc headers on ARM64 | [#5267]    |
| Solidity unavailable: `_BitInt` caps at 128   | [#7344]    |
| Interval analysis differs from x86_64         | [#5267]    |
| CVC5 not built in                             | [#7230]    |
| `--goto-contractor` not built in              | [#7230]    |

Solidity needs `_BitInt(256)` for `uint256`, above what ARM64 Clang supports.

Solidity and interval analysis are the same on macOS Apple Silicon, which is
also ARM64. `--32` is not: that build keeps the bundled 32-bit libc, so the
`--32` tests run there. The two solver gaps are specific to the Linux ARM64
build.

Both Linux assets (x86_64 and ARM64) build against the same LLVM 22, using
trimmed prebuilt archives that contain only the static libraries and headers
ESBMC's static link consumes. The aarch64 archive is produced by a
[reproducible trim script](https://github.com/esbmc/llvm) hosted in the
`esbmc/llvm` repository.

[#5267]: https://github.com/esbmc/esbmc/issues/5267
[#7230]: https://github.com/esbmc/esbmc/issues/7230
[#7344]: https://github.com/esbmc/esbmc/issues/7344
