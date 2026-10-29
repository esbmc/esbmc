// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
// Multi-dim fixed array `uint256[M][N]` round-trip + isolation through
// `__ESOL_deep_copy`.  Fix: the walker uses `_ESBMC_arrcpy_2d` (a single
// C helper call) instead of `c->grid = alloc_array(...)` followed by
// per-slot `c->grid[i] = arrcpy(base->grid[i], ...)`.  The per-slot
// frontend-emitted sequence broke symex value-set tracking (successive
// index writes to a freshly-reassigned pointer field didn't flow
// through to subsequent reads); wrapping the whole allocate+fill dance
// inside one C frame avoids that path.
//
// Historical investigation: `repro_raw/` keeps the raw C / C++
// equivalents that PASS (`raw_u256_c.c`, `raw_u256_cpp.cpp`) plus a
// byte-identical Solidity-emission-pattern C++ repro
// (`raw_u256_cpp_sol_pattern.cpp`).  All three PASS under bitwuzla,
// which was what ruled out cpp_new vs direct struct ASSIGN, `operator=`
// vs direct ASSIGN, and the `_ExtInt(96/160/192)` anon-pad layout as
// hypotheses for the original failure.  The failure was always in the
// frontend's per-slot clone emission — delta-debugging isolated it to
// the `c->grid[i] = arrcpy(...)` writes after the fresh-outer alloc.
function __ESOL_deep_copy(C src) pure returns (C) { return src; }

contract C {
    uint256[2][3] public grid;
    function setAt(uint256 i, uint256 j, uint256 v) public { grid[i][j] = v; }
    function get(uint256 i, uint256 j) public view returns (uint256) { return grid[i][j]; }
}

contract H {
    function check(uint256 a, uint256 b) public {
        require(a != 0 && b != 0 && a != b);
        C base = new C();
        base.setAt(0, 0, a);
        C clone = __ESOL_deep_copy(base);
        // Round-trip: clone sees base's write.
        assert(clone.get(0, 0) == a);
        // Isolation: writes to clone don't propagate to base.
        clone.setAt(0, 0, b);
        assert(base.get(0, 0) == a);
        assert(clone.get(0, 0) == b);
    }
}
