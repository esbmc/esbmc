// SPDX-License-Identifier: MIT
pragma solidity ^0.8.30;

// Dead-arm dual of cov_branch_shortcircuit_pure_pass. Proves the
// short-circuit decision the instrumenter now emits for a folded
// `||` genuinely DISCRIMINATES (it is a real 2-arm decision, not a
// vacuously-100% pair).
//
// `a == a` is a tautology over uint256, so in `(a == a) || (a == 7)`
// the short-circuit decision condition is `a == a`: its true arm is
// reachable but its false arm (`!(a == a)`) is UNSAT. Branch coverage
// is therefore Branches:2 / Reached:1 / 50% — exactly one of the two
// emitted arms is covered. A vacuous/incorrect instrumenter would
// report 100% (or no branch at all). CORE.
library L
{
  function g(uint256 a) internal pure returns (bool)
  {
    return (a == a) || (a == 7);
  }
}
