// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

contract H {
    // Real div-by-zero verification fail: `y` is a public function
    // parameter (symbolic under the dispatcher harness). With default
    // checks (or --div-by-zero-check explicitly), ESBMC's goto-check
    // pass instruments the Yul `div`'s divisor with `assert(y != 0)`
    // and reports "division by zero" when y == 0 is reachable.
    //
    // No manual assert in this contract — the violation is purely from
    // ESBMC's automatic check on the bv division operator. The post-
    // T2.4 lowering exposes the symbolic divisor directly to goto_check
    // (the prior if-guard `if (b==0) 0 else bvudiv(a,b)` made the path
    // condition tautologically imply `b != 0` and silenced the check).
    //
    // Pairs with yul_div_by_zero_pass (require y != 0 → check passes).
    function check(uint256 y) public pure {
        uint256 r;
        assembly {
            r := div(100, y)
        }
    }
}
