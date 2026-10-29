// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
contract B {
    uint256 public y;
    function inner(uint256 v) external { if (v > 50) y = 1; else y = 2; }
}
contract A {
    uint256 public z;
    B b;
    constructor() { b = new B(); }
    function poke(uint256 w) external {
        if (w > 10) z = 1; else z = 2;
        b.inner(w);
    }
}
