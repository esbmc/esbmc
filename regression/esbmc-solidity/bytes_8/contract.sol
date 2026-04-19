// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract ArrayAndBytesBuiltinTest {
    bytes public data;
    bool mutex = false;

    function test() public {
        require(mutex == false);
        data.push();
        data.push(0x12);
        data.push(0x34);
        data.pop();
        assert(data.length == 2);
        mutex = true;
    }
}
