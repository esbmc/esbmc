// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Ensure functions can return different type of value.
contract A {
    function int8_call() public virtual returns (int8) {
        return 21;
    }
    function int16_call() public virtual returns (int16) {
        return 21;
    }
    function int32_call() public virtual returns (int32) {
        return 21;
    }
    function int64_call() public virtual returns (int64) {
        return 21;
    }
    function int128_call() public virtual returns (int128) {
        return 21;
    }
    function int256_call() public virtual returns (int256) {
        return 21;
    }
    function uint8_call() public virtual returns (uint8) {
        return 21;
    }
    function uint16_call() public virtual returns (uint16) {
        return 21;
    }
    function uint32_call() public virtual returns (uint32) {
        return 21;
    }
    function uint64_call() public virtual returns (uint64) {
        return 21;
    }
    function uint128_call() public virtual returns (uint128) {
        return 21;
    }
    function uint256_call() public virtual returns (uint256) {
        return 21;
    }
    function empty() public pure {}
}