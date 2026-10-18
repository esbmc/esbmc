// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Stress: mapping with mixed-shape array value
// `mapping(uint256 => uint8[3][])` — fixed-inner-3 dyn-outer of
// uint8 keyed by uint256.
contract C {
    mapping(uint256 => uint8[3][]) internal records;

    function add(uint256 k, uint8 a, uint8 b, uint8 c) internal {
        uint8[3] memory row;
        row[0] = a;
        row[1] = b;
        row[2] = c;
        records[k].push(row);
    }

    function run() external {
        // initially empty for any key
        assert(records[1].length == 0);
        assert(records[2].length == 0);
        assert(records[42].length == 0);

        // populate key 1
        add(1, 10, 20, 30);
        add(1, 40, 50, 60);
        assert(records[1].length == 2);
        assert(records[1][0][0] == 10);
        assert(records[1][0][1] == 20);
        assert(records[1][0][2] == 30);
        assert(records[1][1][0] == 40);
        assert(records[1][1][1] == 50);
        assert(records[1][1][2] == 60);

        // populate key 2
        add(2, 7, 8, 9);
        assert(records[2].length == 1);
        assert(records[2][0][0] == 7);
        assert(records[2][0][1] == 8);
        assert(records[2][0][2] == 9);

        // key 1 unaffected
        assert(records[1].length == 2);
        assert(records[1][1][2] == 60);

        // pop key 1's last
        records[1].pop();
        assert(records[1].length == 1);
        assert(records[1][0][2] == 30);

        // re-push under key 1 with new values
        add(1, 11, 22, 33);
        assert(records[1].length == 2);
        assert(records[1][1][0] == 11);
        assert(records[1][1][1] == 22);
        assert(records[1][1][2] == 33);

        // mutate first slot of key 1
        records[1][0][1] = 200;
        assert(records[1][0][1] == 200);
        assert(records[1][0][0] == 10);
        assert(records[1][0][2] == 30);

        // key 2 still untouched
        assert(records[2].length == 1);
        assert(records[2][0][1] == 8);
        // key 42 still empty
        assert(records[42].length == 0);
    }
}
