// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Test abi.encodeCall with unbound (nondet) input.
interface ITarget {
    function transfer(uint256 amount) external;
}

contract AbiEncodeCallUnbound {
    function test(uint256 x) public pure {
        uint256 h1 = uint256(keccak256(abi.encodeCall(ITarget.transfer, (x))));
        uint256 h2 = uint256(keccak256(abi.encodeCall(ITarget.transfer, (x))));
        assert(h1 == h2); // deterministic
    }
}
