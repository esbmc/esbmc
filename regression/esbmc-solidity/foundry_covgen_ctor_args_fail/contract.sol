// SPDX-License-Identifier: MIT
pragma solidity ^0.8.30;

// Constructor arguments must be reconstructed: the emitted test builds the
// instance with `new K(<recovered ctor arg>)`, not an uncompilable `new K()`.
contract K {
    uint256 public cap;
    constructor(uint256 c) { cap = c; }
    function use(uint256 v) public {
        if (v < cap) { cap = v; }
    }
}
