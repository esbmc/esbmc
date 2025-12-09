// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract TestAllUnitsAssertionPass {
    function testAllUnits() public pure returns (bool) {
        // Using Solidity literal units
        uint oneWei = 1 wei;
        uint oneGwei = 1 gwei;
        //uint oneSzabo = 1 szabo; // removed in 0.8.0
        //uint oneFinney = 1 finney; // removed in 0.8.0
        uint oneEther = 1 ether;

        // Corresponding values in wei
        assert(oneWei == 1);                            // 1 wei = 1 wei
        assert(oneGwei == 1_000_000_000 wei);           // 1 gwei = 1e9 wei
        //assert(oneSzabo == 1_000_000_000_000 wei);      // 1 szabo = 1e12 wei
        //assert(oneFinney == 1_000_000_000_000_000 wei); // 1 finney = 1e15 wei
        assert(oneEther == 1_000_000_000_000_000_000 wei); // 1 ether = 1e18 wei

        return true;
    }
}