// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// #8661: the narrowing assignment check used the unsigned range for int8.
contract C {
    function f() public pure {
        int8 a = -100;
        a += 10;
        assert(a == -90);
    }
}
