// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

import "./contract2.sol";


contract D is C {
    uint public myDValue = myCValue;

    function test() public view {
        assert(myDValue == 4); 
    }
}
