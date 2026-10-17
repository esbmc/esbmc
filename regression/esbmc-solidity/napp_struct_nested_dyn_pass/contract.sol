// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Stress: struct holding a nested dynamic array `uint256[][]` as a
// member. Exercises field access through struct storage refs +
// nested dyn-array push/pop semantics.
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

        // tag is unaffected by grid mutations
        assert(box.tag == 7);

        // pop inner of row 0; check
        box.grid[0].pop();
        assert(box.grid[0].length == 2);
        assert(box.grid[0][1] == 22);

        // re-push new value
        pushVal(0, 99);
        assert(box.grid[0].length == 3);
        assert(box.grid[0][2] == 99);

        // mutate tag mid-flight
        box.tag = 42;

        // pop outer row
        box.grid.pop();
        assert(box.grid.length == 1);
        assert(box.grid[0].length == 3);
        assert(box.grid[0][0] == 11);
        assert(box.grid[0][1] == 22);
        assert(box.grid[0][2] == 99);
        assert(box.tag == 42);
    }
}
