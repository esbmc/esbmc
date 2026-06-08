// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.8.0 <0.9.0;

contract MixedNestedFail {
    // fixed outer[2], dynamic middle, fixed inner[4]
    uint256[4][][2] internal mixed;

    function run() external {
        // outer fixed length is 2
        // BUG: length is 2 not 3
        assert(mixed.length == 3);
    }
}
