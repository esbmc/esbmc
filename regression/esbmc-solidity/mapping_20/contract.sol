// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Test: large keys (>2^64) are correctly distinguished via XOR-fold hashing
contract LargeKeyMapping {
    mapping(uint256 => uint256) public m;

    function test() public {
        // Two keys that differ only in high bits — would collide with naive truncation
        uint256 k1 = 1;
        uint256 k2 = (1 << 128) + 1;  // differs in bits [128..255]

        m[k1] = 100;
        m[k2] = 200;

        // With XOR-fold: k1 folds to 1, k2 folds to 1^(1<<64) != 1
        // So these should be distinct
        assert(m[k1] == 100);
        assert(m[k2] == 200);
    }
}
