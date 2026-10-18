// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Regression for get_decl_ref_expr asserting current_functionDecl when
// resolving `this`. solc 0.8.x lowers state-variable initializers that
// contain address(this) through a constructor-synthesis path where
// current_functionDecl is not yet set. Before the fix the frontend
// aborted on Converting with an assertion failure.
//
// Assert inside the constructor (where state var init has just run),
// so no multi-tx harness over-approximation is involved.

contract Inner {
    address public parent;
    constructor(address p) { parent = p; }
}

contract C {
    Inner public CHILD;
    address public stored;

    constructor() {
        CHILD = new Inner(address(this));
        stored = address(this);
    }

    function go() external view {
        assert(stored == address(this));
    }
}
