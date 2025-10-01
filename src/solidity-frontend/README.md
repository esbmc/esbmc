
# ESBMC Solidity Front-end

## Overview

The ESBMC Solidity frontend accepts smart contracts written in the Solidity language together with the Solidity AST (in compact JSON form). Based on these inputs, the frontend translates the contract into a C++-style intermediate representation and automatically generates a verification harness (a `main` function). The generated program is then verified by ESBMC's backend.

## Usage

### AST generation

Generate the compact AST JSON using the Solidity compiler (`solc`):

```sh
solc --ast-compact-json example.sol > example.solast
```

The frontend expects the contract source file and its AST JSON as inputs.

### Bug detection

Enable the solidity frontend with the `--sol` option. Example:

```sh
# use --multi-property if you want to report all found bugs at once
esbmc --sol example.sol example.solast --k-induction
```

As with ESBMC in general, ESBMC-Solidity supports common bug classes, including arithmetic overflow/underflow, division by zero, null-dereference, array-out-of-bounds, assertion violations, etc. Use ESBMC command-line options to enable or configure specific checks (e.g. overflow checks, pointer checks, etc.).

Additionally, ESBMC-Solidity provides the `--reentry-check` option to detect potential reentrancy behaviors. When enabled, the frontend inserts assertions and instrumentation to capture possible reentrant call traces. 

Note that a reported reentrancy behavior is an indicator (possible reentrancy path) and does not necessarily imply a concrete exploit (for example, actual loss of funds). The report should be inspected together with the generated counterexample trace.

### Verifying target constraints

By default ESBMC-Solidity analyses every contract present in the provided source program. To restrict verification to a specific contract, use:

```sh
--contract <ContractName>
```

To set multiple contracts:

```sh
--contract "ContractA ContractB"
```

To further restrict the analysis to a particular function, use:

```sh
--contract <ContractName> --function <functionName>
```

`--function` can be used without `--contract`, but this is not recommended because specifying `--contract` disambiguates overloaded / similarly-named functions across contracts. The constraints on contract and function targets help reduce overhead and improve performance.

## Modes

The solidity frontend can construct different `main` harnesses depending on the selected mode. The harness determines how external calls, addresses and inter-contract interactions are modelled, which in turn affects verification purpose and outcomes.

### Unbound reasoning (default)

Flag:

```
--unbound
```

Description: This is the default mode. External calls and addresses that are not otherwise constrained are modelled symbolically — the verifier makes no assumption about the code behind an external (symbolic) address, preserving soundness.

### Bounded cross-contract reasoning

Flag:

```
--bound
```

**Description:** Enabling this mode switches the frontend into *bounded cross-contract reasoning*. In this mode, certain symbolic addresses can be **bound** to known contracts (or to a finite set of contracts). This enables the verifier to model inter-contract calls as concrete jumps to the bound implementations and to generate precise traces that reflect real cross-contract interactions (e.g., attacker contracts interacting with a vulnerable contract or reentrancy scenarios).

This mode is particularly useful for analysing the behaviour of a multi-contract system. Use `--contract` to specify the target contract(s), and `--function` to set the entry point.

## Examples

* Detecting overflow/underflow:

```sh
esbmc --sol MyContract.sol MyContract.solast --overflow-check --unsigned-overflow-check
```

* Running k-induction with solidity frontend:

```sh
esbmc --sol example.sol example.solast --k-induction --multi-property
```

* Bounded analysis :

```sh
esbmc --sol example.sol example.solast --bound --contract Vulnerable --function withdraw
```

## Notes

* During parsing and translation the solidity frontend may introduce additional statements, auxiliary variables and assertion instrumentation. This can affect coverage metrics (e.g. `--branch-coverage`) and the exact locations/format of counterexamples. Use `--goto-functions-only` to inspect the actual intermediate representation the frontend generates.
* The Solidity frontend is still under active development. When encountering unsupported syntax, ESBMC will terminate (and may sometimes crash) during the parsing stage, without proceeding to verification.
* ESBMC's performance is currently affected by the integer bitwidth, especially when `mapping` is involved. For example, `mapping(uint256 => string)` requires more unwinding steps to solve compared to `mapping(uint8 => string)`. You can try the experimental option `--16` to set the machine word size to 16 bits and improve performance.
* Since the minimal supported machine word size is 16, overflow/underflow checks for `uint8` and `int8` are not currently supported.

## For Developers

* The Solidity Frontend convert solidity statement into CPP-style IR -- each contract is considered as a class. Yet no other C++ specific syntax are used, meaning all statements are convert to C-style IR. Data structure like `mapping`, `Bytes` are written as C-struct internally. For more details, check `solidity_template.cpp`.
* The blockchain state is encoded as properties within each contract, such as `address`, `codehash`, `balance`, etc. Note that while in reality a balance is bound to an address, in our modelling it is bound to each contract instance.
* For multi-contract programs, the JSON of each contract is parsed separately. Inheritance is mainly handled by merging the corresponding AST JSON files.