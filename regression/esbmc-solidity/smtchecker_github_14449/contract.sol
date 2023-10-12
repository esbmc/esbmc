// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.0;
contract C
{
     uint256[] y;
    
     function f() public view {
        uint256 x = 0;
         for (uint i = 0; i < y.length; i++) {
             x = 1;
         }
         require(x != 0);
    }
}
