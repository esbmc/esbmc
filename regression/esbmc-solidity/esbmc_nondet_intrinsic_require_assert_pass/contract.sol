// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Dual PASS partner for `esbmc_nondet_intrinsic_require_assert_fail`. The
// require constrains the nondet pair so the downstream equality assert is
// provable. Locks in that `__ESBMC_nondet_uint` is a proper constrainable
// nondet (not unconditionally havoc'd downstream of constraints), and that
// the intrinsic and parameter routes agree on the precise side.
//
// Together with the FAIL partner, this is the dual oracle for the intrinsic.

contract M {
    function __ESBMC_nondet_uint() internal pure returns (uint256) {}
    function __ESBMC_nondet_bool() internal pure returns (bool) {}

    function f() public pure {
        uint256 x = __ESBMC_nondet_uint();
        uint256 y = __ESBMC_nondet_uint();
        bool b = __ESBMC_nondet_bool();
        require(x == y);
        require(b == b); // intrinsic returns a stable value within one call site
        assert(x == y); // provable
    }
}
