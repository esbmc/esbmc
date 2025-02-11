// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.0;

contract test {
    function testArray() public pure{
        uint len = 7;

        uint[] memory a = new uint[](len);
        a[6] = 8;
        assert(a[6] == 8);

        uint[3] memory c = [uint(1), 2, 3];
        assert(c[0] == 1);

        int[] memory ac;
        ac = new int[](10);
        ac[1] = 1;
        assert(ac[1] == 1);
    }
}
