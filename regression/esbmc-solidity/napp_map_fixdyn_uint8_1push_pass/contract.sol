// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Single-push case for `mapping(uint => uint8[3][])`. Pinned by the
// cast fix at solidity_convert_ref.cpp:1141+ (mapping-of-dynarr push
// branch) — the `solidity_gen_typecast(val, base_t.subtype())` at
// line 1165 was bit-reinterpreting the row pointer as a fixed-size
// array, so the local elem_sym held pointer-bits instead of row data.
contract C {
    mapping(uint256 => uint8[3][]) internal records;

    function add(uint256 k, uint8 a) internal {
        uint8[3] memory row;
        row[0] = a;
        records[k].push(row);
    }

    function run() external {
        add(1, 7);
        assert(records[1][0][0] == 7);
    }
}
