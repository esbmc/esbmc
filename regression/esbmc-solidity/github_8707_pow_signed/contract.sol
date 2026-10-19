// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
contract C {
    function f(int256 a, uint256 e) public pure returns (int256) {
        require(a == -2);
        require(e == 3);
        int256 r = a ** e;
        assert(r == -8);
        return r;
    }
}
