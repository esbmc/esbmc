// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Pin: single-modifier + value-returning function (control test for
// the chained-modifier fix in `cov_chained_modifier_return_pass_fix`).
// The single-modifier path has ALWAYS worked because the substitution
// loop sees the original AST body (not a synthesized call) — there is
// no type-nil call expression to misclassify as a bare return.  This
// test pins that base behaviour so a future regression to the fix
// cannot silently break the simpler single-modifier path too.

contract C {
    modifier mod() {
        if (block.timestamp == 0) revert();
        _;
    }
    function f() external mod returns (uint256) {
        return 1;
    }
}
