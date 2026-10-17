// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// T2.1 sub-stage B7-C — bytes-arg distinguishability via fold over
// BytesDynamic.length. abi.encode(uint, bytes) now extracts the bytes
// arg's .length field and folds it into the multi-arg accumulator, so
// two encodings whose bytes args have different lengths produce
// different encoded values and different keccak hashes.
contract H {
    function check(uint256 a, bytes calldata b1, bytes calldata b2) external pure {
        require(b1.length != b2.length);
        bytes32 h1 = keccak256(abi.encode(a, b1));
        bytes32 h2 = keccak256(abi.encode(a, b2));
        assert(h1 != h2);
    }
}
