// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// #8661: a negative int8 result was reported as a narrowing cast overflow.
contract C {
    function f() public pure {
        int8 a = -1;
        int8 b = a + a;
        assert(b == -2);
    }
}
