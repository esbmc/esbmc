// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// FAIL counterpart for `state_var_default_init_no_setter_pass`. Demonstrates the
// recommended pattern for self-composition / adversarial-state queries on a
// contract whose state variables would otherwise stay at default-zero through
// the dispatcher loop (because no public setter exists).
//
// The `__ESBMC_nondet_uint` intrinsic (commit 135c223362) is declared inside
// the contract with an empty body; the Solidity frontend lowers each call to
// `side_effect("nondet", T)` of the AST return type. Assigning the intrinsic
// result into the state variable at function entry havocs the variable for the
// remainder of the trace.
//
// Once `a` and `b` are independent nondets the require `a > b` is satisfiable
// and the downstream `assert(a == b)` fails — the verdict users actually want
// for 2-safety/self-composition oracles.

contract Bug {
    uint256 public a;
    uint256 public b;

    function __ESBMC_nondet_uint() internal pure returns (uint256) {}

    function f() public {
        a = __ESBMC_nondet_uint();
        b = __ESBMC_nondet_uint();
        require(a > b);
        assert(a == b);
    }
}
