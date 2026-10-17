// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.0;

contract CalldataTest {
    function test(bytes calldata x) public returns (bytes calldata) {
        return x;
    }
    function tester(bytes calldata x) public returns (bytes1) {
        require(x.length > 2);
        return this.test(x)[2];
    }
}
