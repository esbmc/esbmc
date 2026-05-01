---
title: Limitations
weight: 4
---

> **Note**: The following limitations apply to the current release of ESBMC-Solidity. Many are actively being addressed. Check the [issue tracker](https://github.com/esbmc/esbmc/issues) for the latest status.

## Inline Assembly (Yul)

Inline assembly blocks are over-approximated: the entire block is replaced by a conservative havoc fallback, which means the surrounding logic may produce spurious counterexamples or miss real ones. Contracts that rely heavily on assembly to enforce invariants are best verified by rewriting the relevant routines in plain Solidity for analysis.

## Dynamic Arrays

- **Multi-call mutation patterns** — `push` and `pop` interactions across many dispatcher iterations may not converge under k-induction. Bounded BMC (`--unwind N`) usually finds bugs in these patterns; full proofs may require restructuring the harness.
- **Symbolic-size arrays** are zero-initialised; the length itself is non-deterministic but constrained to `0 ≤ len ≤ size_t_max` once observed.

## K-Induction

The auto-generated `while(nondet) dispatch()` harness is structurally unbounded — there is no fixed iteration count after which all reachable states are covered.

- **Inductive step coverage gaps** — Some safety properties pass under bounded BMC (`--unwind N`) but cannot be discharged by the inductive step. ESBMC currently does not synthesise loop invariants from contract state, so user-supplied properties that depend on subtle invariants may report `UNKNOWN` under `--k-induction`.
- **Recommendation** — Start with `--unwind N` (or `--incremental-bmc`) for bug-finding, and reach for `--k-induction` only when you need a full correctness proof and you have control over the harness shape.

## ABI Encoding

- `bytesN` (fixed-size byte arrays) and function-reference arguments are folded with reduced precision: ESBMC distinguishes them by length only.
- Total argument width above ~2048 bits triggers a fallback to non-deterministic bytes for the encoded result. The hash collision probability is then formally unconstrained — sound for verification, but assertions that depend on specific encoded bytes may report spurious counterexamples.
- Calldata slicing (`msg.data[i:j]`) before an `abi.*` call is handled under bounded BMC but not under k-induction.

## EOA Balance Model

- `address(addr).balance` reads work only under `--bound`. Without `--bound`, external calls are modelled as arbitrary behaviour and the balance map is unconstrained.
- The balance map uses a linear-scan lookup whose depth is bounded by `--unwind N`. Set `N` to at least the number of distinct EOAs touched on any path; otherwise, balance reads for late-arriving EOAs return non-deterministic values.

## Solver Compatibility

- **Bitwuzla** is the default and is fastest on most workloads.
- Z3, MathSAT, CVC5, and Boolector also work, but performance for Solidity workloads has been less thoroughly tuned.

## Solidity Version Support

- Solidity ≥ **0.5.0** is the minimum.
- Solidity ≥ **0.7** is recommended; ≥ **0.8** is the most thoroughly tested.
- Earlier 0.4.x sources may parse but trigger frontend mismatches around modifier semantics and address-payable distinctions.
