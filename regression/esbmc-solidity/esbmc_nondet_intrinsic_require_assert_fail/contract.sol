// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Mirrors temp_bug.txt MRE (i): `require(a > b); assert(a == b);` where the
// constrained operands are FRESH nondet values. Under the new
// `__ESBMC_nondet_uint` intrinsic, both `x` and `y` are independent nondets,
// so the SMT solver picks a witness with x > y > 0 and the assert FAILs.
//
// Pre-fix (with state variables `a`, `b` instead of intrinsic-sourced
// locals): `require(0 > 0)` pruned the path ⇒ vacuous SUCCESSFUL. The
// intrinsic restores the ability to express adversarial-state queries
// without changing function signatures.

contract M {
    function __ESBMC_nondet_uint() internal pure returns (uint256) {}

    function f() public pure {
        uint256 x = __ESBMC_nondet_uint();
        uint256 y = __ESBMC_nondet_uint();
        require(x > y);
        assert(x == y); // FAIL — independent nondets can satisfy x > y
    }
}
