// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Dual-fail: struct.grid same harness as PASS, flipped tag invariant
// after the second tag mutation.
contract C {
    struct Box {
        uint256 tag;
        uint256[][] grid;
    }

    Box internal box;

    function pushRow() internal {
        box.grid.push();
    }

    function pushVal(uint256 r, uint256 v) internal {
        box.grid[r].push(v);
    }

    function run() external {
        box.tag = 7;
        assert(box.tag == 7);
        assert(box.grid.length == 0);

        pushRow();
        pushRow();
        assert(box.grid.length == 2);
        assert(box.grid[0].length == 0);
        assert(box.grid[1].length == 0);

        pushVal(0, 11);
        pushVal(0, 22);
        pushVal(0, 33);
        assert(box.grid[0].length == 3);
        assert(box.grid[0][0] == 11);
        assert(box.grid[0][1] == 22);
        assert(box.grid[0][2] == 33);

        pushVal(1, 100);
        assert(box.grid[1].length == 1);
        assert(box.grid[1][0] == 100);

        assert(box.tag == 7);

        box.grid[0].pop();
        assert(box.grid[0].length == 2);
        assert(box.grid[0][1] == 22);

        pushVal(0, 99);
        assert(box.grid[0].length == 3);
        assert(box.grid[0][2] == 99);

        box.tag = 42;

        box.grid.pop();
        assert(box.grid.length == 1);
        // FLIPPED: tag is 42 after the second assignment, not 7
        assert(box.tag == 7);
        assert(box.grid[0].length == 3);
        assert(box.grid[0][0] == 11);
        assert(box.grid[0][1] == 22);
        assert(box.grid[0][2] == 99);

        // mutate tag again — and re-mutate grid
        box.tag = 5;
        assert(box.tag == 5);
        box.grid[0].pop();
        assert(box.grid[0].length == 2);
    }
}
