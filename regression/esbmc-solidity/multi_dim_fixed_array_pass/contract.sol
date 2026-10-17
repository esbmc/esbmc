// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.8.0 <0.9.0;

// Round-trip test for fixed multi-dim arrays `uint256[N][M]`.
// Under correct semantics the assertions all hold (we read back what we wrote).
//
// Currently KNOWNBUG: the frontend's `get_array_size()` regex only captures
// the last `[N]` from the typeString, so `uint256[3][2]` collapses to a 1-D
// array and writes to the outer row are lost. See
// CLAUDE_Solidity.md §B and docs/claude/solidity/language-support.md §B.
contract MultiDimFixedPass {
    // Solidity convention: `uint256[3][2]` = 2 rows, each holding 3 uints.
    uint256[3][2] public grid;

    function check() external {
        grid[0][0] = 10;
        grid[0][1] = 20;
        grid[0][2] = 30;
        grid[1][0] = 40;
        grid[1][1] = 50;
        grid[1][2] = 60;

        assert(grid[0][0] == 10);
        assert(grid[0][1] == 20);
        assert(grid[0][2] == 30);
        assert(grid[1][0] == 40);
        assert(grid[1][1] == 50);
        assert(grid[1][2] == 60);
    }
}
