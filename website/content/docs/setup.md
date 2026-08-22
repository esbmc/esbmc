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
| Solidity unavailable: `_BitInt` caps at 128   | [#5267]    |
| Interval analysis differs from x86_64         | [#5267]    |
| CVC5 not built in                             | [#7230]    |
| `--goto-contractor` not built in              | [#7230]    |

Solidity needs `_BitInt(256)` for `uint256`, above what ARM64 Clang supports.

The first three apply to macOS Apple Silicon too; the two solver gaps are
specific to the Linux ARM64 build.

#### Why the ARM64 build uses LLVM 18

The x86_64 build links a prebuilt LLVM 22 that the project hosts itself, trimmed
to about 140MB. No equivalent exists for aarch64: LLVM stopped publishing the
compact `clang+llvm-*-aarch64-linux-gnu` archive after 19.x, and the
`LLVM-*-Linux-ARM64` archive that replaced it unpacks to 11GB, more than a
public ARM CI runner's disk. The ARM64 build therefore links the distribution's
own LLVM packages, and Ubuntu 24.04 ships LLVM 18.

LLVM 18 is the minimum ESBMC supports, so this is a supported configuration
rather than a downgrade, and the binary is still statically linked with no
runtime dependency on LLVM. Verification results do not depend on the LLVM
version: it supplies the C/C++ parser, not the solver.

This resolves once a trimmed aarch64 LLVM 22 archive is hosted alongside the
x86_64 one, tracked in [#7236].

[#5267]: https://github.com/esbmc/esbmc/issues/5267
[#7230]: https://github.com/esbmc/esbmc/issues/7230
[#7236]: https://github.com/esbmc/esbmc/issues/7236
