// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Test abi.encodeCall via interface function reference with bound input.
// Note: abi.encodeCall requires a function pointer; we use an interface.
interface ITarget {
    function transfer(uint256 amount) external;
}

contract AbiEncodeCallBound {
    function test() public pure {
        uint256 h1 = uint256(keccak256(abi.encodeCall(ITarget.transfer, (uint256(42)))));
        uint256 h2 = uint256(keccak256(abi.encodeCall(ITarget.transfer, (uint256(42)))));
        assert(h1 == h2); // deterministic
    }
}
