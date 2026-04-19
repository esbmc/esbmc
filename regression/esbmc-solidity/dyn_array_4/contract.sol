// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.0;

contract test {
    int16[] public data1;

    function testArray() public {

        int16[] memory ac;
        ac = new int16[](10);
        ac[1] = 1;
        data1 = ac;
        
        data1 = [int16(-60), 70, -80, 90, -100, -120, 140];
        assert(data1[1] == 1);
    }
}
