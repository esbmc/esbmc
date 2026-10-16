// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract C {
    function pair(bool choose) internal pure returns (uint256 a, uint256 b) {
        if (choose) {
            a = 7;
        } else {
            a = 11;
        }
        b = a + 1;
    }

    function check(bool choose) public pure {
        (uint256 a, uint256 b) = pair(choose);
        assert(a == (choose ? 7 : 11) && b == a + 1);
    }
}
