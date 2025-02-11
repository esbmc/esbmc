// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.0;

// Creating a contract
contract Base {
    function test() external pure {
        {
            uint same;
            same = 1;
        }

        {
            uint same = 0;
            assert(same == 0);
        }
    }
}