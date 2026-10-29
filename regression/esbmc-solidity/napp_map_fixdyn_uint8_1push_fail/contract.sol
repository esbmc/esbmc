// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Companion FAIL test for napp_map_fixdyn_uint8_1push_pass.
contract C {
    mapping(uint256 => uint8[3][]) internal records;

    function add(uint256 k, uint8 a) internal {
        uint8[3] memory row;
        row[0] = a;
        records[k].push(row);
    }

    function run() external {
        add(1, 7);
        assert(records[1][0][0] == 8);
    }
}
