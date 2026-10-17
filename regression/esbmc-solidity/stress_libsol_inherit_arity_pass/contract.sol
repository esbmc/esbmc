// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Regression for the base-ctor arity-mismatch fix in
// get_inherit_static_contract_instance. `Middle` is abstract and does
// NOT forward `Root`'s ctor args in its own modifier list; only the
// concrete descendant `C` supplies them. Without the fix, the
// synthesized `_ESBMC_aux_Root` copy helper inside Middle's ctor was
// emitted with `(this, aux_bool)` against a formal of
// `(this, uint256, uint256, aux_bool)`, tripping symex's
// "function call: not enough arguments" before go() could run.

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
    function go() external view returns (uint256) {
        return a + b + m;
    }
}
