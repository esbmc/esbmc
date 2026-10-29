// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.8.0 <0.9.0;

// 2D fully-fixed `uint256[N][M]` as a function parameter. Currently
// ESBMC's Solidity harness synthesises fixed-array parameters via
// calloc'd backing storage (see solidity_convert_call.cpp, 2D branch
// at ~line 680). Native nested array_typet params may not yet be on
// that path. This test documents the round-trip expectation on the
// param-based pattern; if the frontend binds native-nested directly
// the harness must still ensure callee reads see writer-initialised
// values.
contract MultiDimFnParam2DPass {
    function compute(uint256[3][2] memory grid) external pure returns (uint256) {
        // Sum diagonal; caller is the harness — over-approximation
        // means grid's content is whatever the backing buffer was
        // initialised to. We only assert that the callee's OWN writes
        // round-trip within the body.
        uint256 a = grid[0][0];
        uint256 b = grid[1][1];
        grid[0][0] = 100;
        grid[1][1] = 200;
        assert(grid[0][0] == 100);
        assert(grid[1][1] == 200);
        return a + b;
    }
}
