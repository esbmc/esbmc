// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Dual to pass: ensure the address(this) path in a state-variable
// initializer actually reaches symex, using a deliberately-wrong
// post-constructor check.

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
        assert(stored == address(0)); // wrong
    }
}
