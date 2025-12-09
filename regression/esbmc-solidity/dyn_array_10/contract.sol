// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.0;

contract test {
    int16[] public data1;

    function testArray() public {
        assert(data1.length == 0);
        data1.push();

    }
}
