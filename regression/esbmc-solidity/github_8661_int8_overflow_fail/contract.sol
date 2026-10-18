// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// #8661: 127 + 1 does not fit in an int8 and was not reported.
contract C {
    function f() public pure {
        int8 a = 127;
        int8 b = a + 1;
    }
}
