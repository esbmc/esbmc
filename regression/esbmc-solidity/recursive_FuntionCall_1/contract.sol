// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract C {
    uint x;

    function f() internal {
        x += 1;
        assert(x < 5);  
        f();            
    }

    constructor() {
        f();
    }
}
