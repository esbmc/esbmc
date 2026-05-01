// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.8.0 <0.9.0;

contract FixedDynFail {
    uint256[][3] internal a3;

    function run() external {
        // outer fixed length is 3
        // BUG: a3.length is 3, not 4
        assert(a3.length == 4);
    }
}
