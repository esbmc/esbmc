// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;
contract C {
    function f(uint256[3][2] calldata s) external pure returns (uint256) {
        return s[1][2];
    }
}
