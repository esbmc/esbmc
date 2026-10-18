// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;
contract C {
    function f(uint256[] calldata a) external pure returns (uint256) {
        return a[0];
    }
}
