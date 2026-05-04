// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Test abi.encodeCall: different arguments must produce different encoding.
interface ITarget {
    function transfer(uint256 amount) external;
}

contract AbiEncodeCallDiff {
    function test() public pure {
        uint256 h1 = uint256(keccak256(abi.encodeCall(ITarget.transfer, (uint256(1)))));
        uint256 h2 = uint256(keccak256(abi.encodeCall(ITarget.transfer, (uint256(2)))));
        assert(h1 == h2); // must FAIL: different arguments
    }
}
