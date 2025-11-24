// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

library CompareLib {
    function isGreater(uint256 x, uint256 y) internal pure returns (bool) {
        return x > y;
    }
    function isLess(uint256 x, uint256 y) internal pure returns (bool) {
        return x < y;
    }   
}

contract Comparator {
    function testCompare() public pure {
        bool result = CompareLib.isGreater(10, 5);
        assert(result); 
    }
}
