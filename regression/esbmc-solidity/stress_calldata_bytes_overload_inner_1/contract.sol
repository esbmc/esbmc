// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.0;

contract C {
    function f(bytes calldata b, uint i) internal pure returns (bytes1) {
        return b[i];
    }
    function f(uint, bytes calldata b, uint) external pure returns (bytes1) {
        require(b.length > 2);
        return f(b, 2);
    }
}
