// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Test abi.encodePacked: string equality via keccak256 hash comparison.
// This is the common Solidity pattern for comparing strings.
contract AbiEncodePackedStringEq {
    function test() public pure {
        string memory s1 = "hello";
        string memory s2 = "hello";
        assert(
            keccak256(abi.encodePacked(s1)) == keccak256(abi.encodePacked(s2))
        ); // must hold: same string content
    }
}
