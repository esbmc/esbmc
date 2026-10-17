// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Dual-fail: identical 2D-dyn int param harness; sumAndCheck is
// called once with a deliberately wrong expected value.
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
        assert(grid.length == 3);
        assert(grid[0].length == 4);
        assert(grid[1].length == 2);
        assert(grid[2].length == 0);
        assert(grid[0][0] == 1);
        assert(grid[0][1] == -2);
        assert(grid[0][2] == 3);
        assert(grid[0][3] == -4);
        assert(grid[1][0] == 100);
        assert(grid[1][1] == -100);
    }

    function run() external {
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

        lengthCheck(grid);

        // FLIPPED: actual sum is -2, not 99
        int256 result = sumAndCheck(grid, 99);
        accumulator = result;
        assert(accumulator == -2);

        // additional mutation paths to mirror PASS structure
        grid[0][3] = 0;
        result = sumAndCheck(grid, 2);
        assert(result == 2);

        grid[1][0] = -100;
        grid[1][1] = 100;
        result = sumAndCheck(grid, 2);
        assert(result == 2);

        accumulator = result;
        assert(accumulator == 2);

        // build a fresh smaller grid and run sumAndCheck again
        int256[][] memory g2 = new int256[][](2);
        g2[0] = new int256[](2);
        g2[0][0] = 7;
        g2[0][1] = -3;
        g2[1] = new int256[](1);
        g2[1][0] = 11;
        result = sumAndCheck(g2, 15);
        assert(result == 15);
    }
}
