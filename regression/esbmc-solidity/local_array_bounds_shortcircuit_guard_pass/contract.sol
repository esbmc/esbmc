// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// No bounds-check false positive when a pointer-array access sits behind a
// short-circuit guard: `k < 2 && b[k] == 0` never evaluates b[k] for k >= 2.
// `&&`/`||` lower to `and`/`or`, whose operands GOTO conversion emits under the
// short-circuit guard, so the bounds assertion queued for b[k] is only reachable
// when k < 2 holds. k-induction therefore proves this SUCCESSFUL rather than
// reporting a spurious array-bounds violation.
contract C {
    function f(uint k) public pure {
        uint[] memory b = new uint[](2);
        if (k < 2 && b[k] == 0) { assert(true); }
    }
}
