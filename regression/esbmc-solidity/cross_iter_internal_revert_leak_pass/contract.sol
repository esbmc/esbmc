// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// B5-B Stage S0 — empirically test whether internal-helper revert leaks
// state across harness iterations.
// Iteration N: tryWrite() calls _helper() which writes x=1 then reverts
//              via require(false). Internal helpers use legacy
//              __ESBMC_assume(false) — no B1 rollback.
// Iteration N+1: check() asserts x == 0. If the SSA carries the pruned-
//              path write into the surviving sibling-iteration branch,
//              this assertion fails (real bug). If the SSA branching at
//              the while-loop join correctly predicates writes by the
//              feasibility of the path that produced them, the
//              assertion holds (already covered).
contract H {
    uint256 public x;

    function _helper() internal {
        x = 1;
        require(false, "revert");
    }

    function tryWrite() external {
        _helper();
    }

    function check() external view {
        assert(x == 0);
    }
}
