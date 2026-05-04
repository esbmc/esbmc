// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0 <0.9.0;

contract MapArrayTest {
    mapping(uint => uint)[] array;

    function test() public {
        array.push();
        array[0][1] = 42;
        assert(array[0][1] == 42);
    }
}
