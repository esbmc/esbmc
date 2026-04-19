// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

library CheckLib {
    function isEven(uint256 number) internal pure returns (bool) {
        return number % 2 == 0;
    }
}

contract Checker {
    function checkEven(uint256 number) public pure {
        bool even = CheckLib.isEven(number);
        assert(even == true); 
    }
}