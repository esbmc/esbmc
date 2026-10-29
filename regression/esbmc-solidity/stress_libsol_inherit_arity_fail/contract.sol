// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Dual to the pass test: same abstract-parent inheritance pattern that
// exercises get_inherit_static_contract_instance's nondet-pad path, but
// go() carries an unconditional `assert(false)`. Without the base-ctor
// arity fix the frontend crashes with
// "function call: not enough arguments" before symex reaches go(); with
// the fix, symex reaches the body and reports the seeded violation as
// VERIFICATION FAILED.

contract Root {
    uint256 public a;
    uint256 public b;
    constructor(uint256 _a, uint256 _b) {
        a = _a;
        b = _b;
    }
}

abstract contract Middle is Root {
    uint256 public m;
    constructor(uint256 _m) {
        m = _m;
    }
}

contract C is Middle {
    constructor() Root(1, 2) Middle(3) {}
    function go() external view {
        assert(false);
    }
}
