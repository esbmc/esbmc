// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// #8661: 100 + 28 does not fit in an int8 and was not reported.
contract C {
    function f() public pure {
        int8 a = 100;
        a += 28;
    }
}
