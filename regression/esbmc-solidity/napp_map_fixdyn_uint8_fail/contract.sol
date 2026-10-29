// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Dual-fail: same mapping(uint => uint8[3][]) harness with flipped
// inner-mutation invariant.
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
        assert(records[1].length == 0);
        assert(records[2].length == 0);
        assert(records[42].length == 0);

        add(1, 10, 20, 30);
        add(1, 40, 50, 60);
        assert(records[1].length == 2);
        assert(records[1][0][0] == 10);
        assert(records[1][0][1] == 20);
        assert(records[1][0][2] == 30);
        assert(records[1][1][0] == 40);
        assert(records[1][1][1] == 50);
        assert(records[1][1][2] == 60);

        add(2, 7, 8, 9);
        assert(records[2].length == 1);
        assert(records[2][0][0] == 7);
        assert(records[2][0][1] == 8);
        assert(records[2][0][2] == 9);

        assert(records[1].length == 2);

        records[1].pop();
        assert(records[1].length == 1);
        assert(records[1][0][2] == 30);

        add(1, 11, 22, 33);
        assert(records[1].length == 2);
        assert(records[1][1][0] == 11);
        assert(records[1][1][1] == 22);
        assert(records[1][1][2] == 33);

        records[1][0][1] = 200;
        // FLIPPED: records[1][0][1] is 200 after the write, not 20
        assert(records[1][0][1] == 20);
        assert(records[1].length == 2);
        assert(records[1][0][0] == 10);
        assert(records[1][0][2] == 30);

        assert(records[2].length == 1);
        assert(records[2][0][0] == 7);
        assert(records[2][0][1] == 8);
        assert(records[2][0][2] == 9);
        assert(records[42].length == 0);

        // additional mutations on key 2
        add(2, 50, 60, 70);
        assert(records[2].length == 2);
        assert(records[2][1][0] == 50);
        assert(records[2][1][1] == 60);
        assert(records[2][1][2] == 70);
    }
}
