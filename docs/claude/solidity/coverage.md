# Solidity Coverage Support

This doc captures the Solidity-specific handling of coverage.

## Quick Reference

```bash
# Branch coverage (use --focus-function for targeted analysis)
esbmc contract.sol --contract MyContract --focus-function myFunc \
  --branch-coverage-claims --unwind 10 --no-unwinding-assertions

# Condition coverage (whole-contract)
esbmc contract.sol --contract MyContract \
  --condition-coverage-claims --unwind 10 --no-unwinding-assertions

# Assertion coverage
esbmc contract.sol --contract MyContract --focus-function myFunc \
  --assertion-coverage-claims --unwind 10 --no-unwinding-assertions

# Branch coverage over the whole contract (auto-skips multi-tx harness)
esbmc contract.sol --contract MyContract --branch-coverage \
  --unwind 10 --no-unwinding-assertions
```

## Multi-Transaction Harness Neutralization

The Solidity frontend generates `_ESBMC_Main_*` harness functions containing a `while(nondet_bool())` loop for multi-transaction verification. In coverage mode, this loop causes massive symbolic execution overhead since it repeatedly calls user functions.

**Fix** (`esbmc_parseoptions.cpp`): When Solidity + coverage mode is detected, backward GOTOs (loop back-edges) in `_ESBMC_Main*` functions are converted to SKIPs before symbolic execution begins. This makes the harness execute each user function exactly once, which is sufficient for coverage analysis.

Transparent to the user — no extra flags needed. Normal (non-coverage) verification is unaffected.

## Modifier-Expanded Function Matching

Solidity modifiers rename functions: `deposit` with modifier `onlyPositive` becomes `deposit_onlyPositive`. The single-function target supports prefix matching for Solidity so that targeting `deposit` matches `deposit_onlyPositive`.

Implementation: `goto_coverage.cpp:is_target_func()`. For `language_idt::SOLIDITY`, if the symbol name starts with `tgt_name + "_"`, it is considered a match.

## Expression Pretty-Printing

Coverage reports automatically prettify C-level expressions to Solidity-friendly notation when `config.language.lid == SOLIDITY`:

| C-level | Solidity |
|---------|----------|
| `(signed int)y < 50` | `y < 50` |
| `(unsigned _ExtInt(256))msg_sender` | `msg.sender` |
| `this->owner` | `owner` |
| `balances[(signed long int)((unsigned _ExtInt(256))msg_sender)]` | `balances[msg.sender]` |
| `msg_value`, `tx_origin`, `block_number`, etc. | `msg.value`, `tx.origin`, `block.number`, etc. |

Implementation: `bmc.cpp:prettify_solidity_expr()`. Applied at display time only — claim matching logic uses the original C-level strings.

## Filter Mechanism

Two-stage filtering ensures only user-written Solidity code is instrumented:

1. **Function level** (`filter()`): Skips functions marked with `__ESBMC_HIDE` label. The Solidity frontend marks ~20+ auxiliary functions (constructors, initializers, dispatchers, mapping helpers) with this label.
2. **Instruction level** (`location_pool`): Only instruments code from `.sol` source files. Library code from `sol64.goto` has `.c` file locations and is automatically excluded.

## Solidity-Specific Behaviour

- `require(cond)` → modeled as `__ESBMC_assume(cond)` → **not counted** in branch coverage (correctly, since require is not a branch)
- `assert(cond)` → counted in assertion coverage; replaced with `assert(true)` in branch/condition coverage
- Overflow checks → inserted as assertions by `goto_check`, replaced with `assert(true)` in branch coverage → **not counted** as branches
- `unchecked { }` blocks → overflow assertions already suppressed, no impact on coverage

## Zero-Goals Coverage Summary

When a program has 0 instrumented coverage goals (e.g., straight-line code with no branches), the `[Coverage]` summary section is still printed:

```
[Coverage]

Branches : 0
Reached : 0
Branch Coverage: N/A (no branches)
```

## Future Work (Solidity-specific)

**Already works (no changes needed):**
- `--cov-report-json` — JSON report generation is language-agnostic, uses standard location format
- `scripts/cov-report.py` — HTML report generator reads JSON, works with any source file including `.sol`
- Counterexample traces — built from SSA steps, language-agnostic

**Needs new code:**
- Solidity testcase generator — new `solidity_testcase_generator` class (~2000-3000 lines). Current `pytest_generator` and `ctest_generator` are Python/C-specific. Solidity would need: uint256/address/bytes32 type mapping, contract state initialization, ABI encoding, and choice of test framework (Hardhat/Foundry).
