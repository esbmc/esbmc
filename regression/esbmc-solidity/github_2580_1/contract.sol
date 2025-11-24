// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

//!BUG: k-induction will lead to internal checks fail!!!
contract ArrayAndBytesBuiltinTest {
    uint[] public numbers;

    function test() public {
        numbers.push(10);
        numbers.push(20);
        numbers.pop();
        assert(numbers.length == 1);
    }
}
