// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
contract C {
    function f() public pure returns (int256) {
        int256 m = type(int256).min;
        assert(m < 0);
        return m;
    }
}
