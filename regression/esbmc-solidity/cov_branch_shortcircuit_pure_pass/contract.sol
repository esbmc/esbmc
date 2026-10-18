// SPDX-License-Identifier: MIT
pragma solidity ^0.8.30;

// Minimal standalone repro of the pure short-circuit branch-coverage
// gap surfaced by limit-order-protocol/MakerTraitsLib.isAllowedSender
// (TWO_TRACK_AGGREGATE.md finding 3).
//
// `a == 0 || b == 1` has side-effect-free operands, so the frontend
// folds the `||` into a flat boolean expression inside a single RETURN
// (no GOTO split). Before the fix, goto_coverage.cpp::branch_coverage
// only instrumented `it->is_goto()` control-flow guards, so this
// short-circuit decision was invisible -> ESBMC emitted
// "No branch detected", while solc instruments every `||`/`&&` as a
// 2-arm branch decision (cf. MakerTraitsLib.useBitInvalidator, whose
// `||` operands are function calls -> GOTO split -> Branches:2).
//
// FIXED: collect_short_circuit_decisions walks ASSIGN/RETURN exprs and
// emits the same 2-arm decision for folded or2t/and2t. ESBMC now
// reports `Branches : 2 / 100%` here, at parity with solc. CORE; the
// dead-arm dual is cov_branch_shortcircuit_partial_pass (50%).
library L
{
  function f(uint256 a, uint256 b) internal pure returns (bool)
  {
    return a == 0 || b == 1;
  }
}
