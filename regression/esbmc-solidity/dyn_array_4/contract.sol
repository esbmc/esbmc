// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.0;

contract test {
    int[] public data1;

    function testArray() public {

        int[] memory ac;
        ac = new int[](10);
        ac[1] = 1;
        data1 = ac;
        
        data1 = [int(-60), 70, -80, 90, -100, -120, 140];
        assert(data1[1] == 1);
    }
}
