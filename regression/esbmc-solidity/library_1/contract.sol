// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

library MathLib {
    function add(uint8 a, uint8 b) internal pure returns (uint256) {
        return a + b;
    }

}

contract Calculator {
    uint8 public a = 5;
    uint8 public b = 1;
    uint256 public result; 
    function sum() public {
        result = MathLib.add(a, b);
        assert(result == 15);
    }
}