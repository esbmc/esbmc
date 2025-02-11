// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.0;

contract test {
    uint[3] public t;
    uint[] public tt;

    function testArray() public {
        t = [1, 2, 3];
        assert(t[1] == 2);
        tt = [4, 5, 6];
        assert(tt[1] + t[0] == tt[2]);
    }
}
