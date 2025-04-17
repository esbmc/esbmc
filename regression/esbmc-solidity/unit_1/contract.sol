// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract TestAssertionPass {
    function testUnits() public pure returns (bool) {
        // Using solidity units:
        uint valueWei = 1 wei;       // 1 wei
        uint valueGwei = 1 gwei;     // 1 gwei
        
        // Assertion: 1 gwei should equal 1,000,000,000 wei.
        assert(valueGwei == 1000000000 * valueWei);
        
        return true;
    }
}
