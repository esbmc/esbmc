// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract UnderflowWithUnits {
    function testUnderflowEtherUnits() public pure returns (uint256) {
        uint256 balance = 0;
        uint256 result = balance - 1 gwei; // underflow here
        return result;
    }
}