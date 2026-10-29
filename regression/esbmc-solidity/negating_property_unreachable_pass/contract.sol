// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
contract C {
    function f(uint256 x) public pure {
        if (x > x) {
            assert(1 == 1);
        }
    }
}
