// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract UnitTest {
    uint public oneEther = 1 ether;
    uint public tenGwei = 10 gwei;
    uint public someWei = 500 wei;

    uint public duration = 5 minutes;

    function convertToWei(uint _eth) public pure returns (uint) {
        return _eth * 1 ether;
    }

    function waitTime() public pure returns (uint) {
        return 2 hours + 30 minutes;
    }
}