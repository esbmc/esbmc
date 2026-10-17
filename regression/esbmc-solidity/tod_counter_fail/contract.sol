// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// add() and mul() both touch x, so swapping them produces different x.
// The TOD harness should detect this.
contract Counter {
    uint public x;

    constructor() {
        x = 0;
    }

    function add(uint n) public {
        x = x + n;
    }

    function mul(uint n) public {
        x = x * n;
    }
}
