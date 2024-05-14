// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.0;

// Creating a contract
contract Base {
    function test() external pure{
        {
            uint same;
            same = 1;
        }

        {
            // default val  = 0
            uint same;
            assert(same == 0);
        }
    }
}
