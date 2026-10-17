// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.8.0 <0.9.0;

// Violation test for fixed multi-dim arrays `uint256[N][M]`.
// Under correct semantics writing `grid[1][0] = 10` leaves `grid[0][0]`
// holding the earlier write of 5, so `assert(grid[0][0] != 5)` must fail.
//
// Currently KNOWNBUG: the frontend collapses `uint256[3][2]` to a 1-D
// array and the writes are lost/aliased, so the assertion spuriously
// succeeds. See CLAUDE_Solidity.md §B and
// docs/claude/solidity/language-support.md §B.
contract MultiDimFixedFail {
    uint256[3][2] public grid;

    function check() external {
        grid[0][0] = 5;
        grid[1][0] = 10;
        assert(grid[0][0] != 5);
    }
}
