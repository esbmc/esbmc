# ESBMC Solidity Front-end

## Overview

ESBMC Frontend accept Smart Contract program written in Solidity language. Apart from the contract, its ast json should also input together. Base on this information, the frontend will translate the contract into a CPP-style intermediate representation, with a automatically generated harness as "main" function. This eventually verifiered by ESBMC's backend.

## Usage

### AST Generation

The generation of the AST Json requires the Solidity Compiler (solc) via command:

```sh
solc --ast-compact-json example.sol > example.solast
```

### Bug Detection

The esbmc-solidity frontend is enabled by option `--sol`. E.g.

```sh
# use --multi-property if you want to report all the bugs in one times
esbmc --sol example.sol example.solast --k-induction 
```

Just like ESBMC, ESBMC-Solidity supports all common bug types, including: Arithmatic Overflow/Underflow, Diviison by zero, et, al. You can use the corresponding options to enable the checkings.

Additionally, ESBMC-Solidity provides option `--reentry-check` to detect the reentrancy behaviour, that is .... . By enabling this option, the solidity frontend will insert assertion instrumentation to capture the any possible reentrancy behaviour. Note that this behaviour might not neccesary leads to an actual reentrancy exploit (E.g. balance loss). 

### Verifying Target Constraints

By default, ESBMC-Solidity will analyse all the contracts included in the source program. To limit the verification tartget, use option `--contract X`, where `X` is the name of the contract. Multiple contract can be set via `--contract "X Y"`.

If the target is further constraint to a funciton, use options `--contract X --function x`, where `x` is the name of the function. `--function` can be used without `--contract` but is not recommended.  

## Mode

Depending on the mode selected, the solidity frontend will construct difference "main" harness, which will affect the verification purpose, modelling and outcomes. 

- `--unbound`: enable by default. 

- `--bound`: enable this will overwrite the default unbound setting.

## Note 

- 