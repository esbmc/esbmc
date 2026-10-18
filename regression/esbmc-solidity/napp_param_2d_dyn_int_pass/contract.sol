// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Stress: nested array as a memory function parameter — `int256[][]`.
// Tests that signed-element 2D dynamic arrays round-trip correctly
// across a memory parameter boundary, and that .length and indexed
// reads on inner rows are consistent with what the caller built.
contract C {
    int256 internal accumulator;

    function sumAndCheck(int256[][] memory grid, int256 expected)
        internal
        returns (int256)
    {
        int256 acc = 0;
        for (uint256 i = 0; i < grid.length; i++) {
            for (uint256 j = 0; j < grid[i].length; j++) {
                acc += grid[i][j];
            }
        }
        assert(acc == expected);
        return acc;
    }

    function lengthCheck(int256[][] memory grid) internal pure {
        // grid has 3 outer rows in our caller
        assert(grid.length == 3);
        assert(grid[0].length == 4);
        assert(grid[1].length == 2);
        assert(grid[2].length == 0);
        // indexed reads on row 0
        assert(grid[0][0] == 1);
        assert(grid[0][1] == -2);
        assert(grid[0][2] == 3);
        assert(grid[0][3] == -4);
        // indexed reads on row 1
        assert(grid[1][0] == 100);
        assert(grid[1][1] == -100);
    }

    function run() external {
        // build a 3-row jagged grid in memory: row 0 has 4, row 1 has 2,
        // row 2 is empty
        int256[][] memory grid = new int256[][](3);
        grid[0] = new int256[](4);
        grid[0][0] = 1;
        grid[0][1] = -2;
        grid[0][2] = 3;
        grid[0][3] = -4;
        grid[1] = new int256[](2);
        grid[1][0] = 100;
        grid[1][1] = -100;
        grid[2] = new int256[](0);

        // expected sum = 1 - 2 + 3 - 4 + 100 - 100 = -2
        lengthCheck(grid);
        int256 result = sumAndCheck(grid, -2);
        assert(result == -2);

        // mutate one row in memory; sum changes
        grid[0][3] = 0;
        // new sum = 1 - 2 + 3 + 0 + 100 - 100 = 2
        result = sumAndCheck(grid, 2);
        assert(result == 2);

        // re-pass with reversed inner row 1 — shouldn't change sum
        grid[1][0] = -100;
        grid[1][1] = 100;
        result = sumAndCheck(grid, 2);
        assert(result == 2);

        accumulator = result;
        assert(accumulator == 2);
    }
}
