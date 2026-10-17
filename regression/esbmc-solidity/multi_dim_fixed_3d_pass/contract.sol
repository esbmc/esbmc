// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.8.0 <0.9.0;

// 3D fully-fixed array round-trip: all three dims are compile-time
// constants, so the native `array_typet(array_typet(array_typet(T, 2),
// 3), 4)` path from option B (commit c5eec55601) applies recursively.
//
// Exercises element round-trip over all three index levels and
// cross-row isolation (writes to one slice don't bleed into others).
// Does NOT rely on zero-init default — the harness re-enters `run()`
// non-deterministically, so prior-call state means `g[i][j][k]` may be
// any user-written value on re-entry. `pin()` collapses to a fixed
// state before the round-trip so the assertions are deterministic.
contract MultiDimFixed3DPass {
    uint256[2][3][4] internal g;

    function pin() internal {
        g[0][0][0] = 100;
        g[1][1][1] = 200;
        g[2][2][0] = 300;
        g[3][0][1] = 400;
        // Unrelated cells (one per outer slice) explicitly zeroed so
        // the isolation checks below are sound regardless of prior
        // calls to run().
        g[0][0][1] = 0;
        g[1][0][0] = 0;
        g[2][1][1] = 0;
        g[3][1][0] = 0;
    }

    function run() external {
        pin();

        // Diagonal reads return the written values.
        assert(g[0][0][0] == 100);
        assert(g[1][1][1] == 200);
        assert(g[2][2][0] == 300);
        assert(g[3][0][1] == 400);

        // Unrelated cells remain zero (cross-row isolation).
        assert(g[0][0][1] == 0);
        assert(g[1][0][0] == 0);
        assert(g[2][1][1] == 0);
        assert(g[3][1][0] == 0);

        // Overwrite preserves other cells.
        g[0][0][0] = 999;
        assert(g[0][0][0] == 999);
        assert(g[1][1][1] == 200);
        assert(g[2][2][0] == 300);
    }
}
