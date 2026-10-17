// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
// 3D fixed-array `uint256[M][N][K]` through `__ESOL_deep_copy`.
//
// Was KNOWNBUG — the walker emits (A) outer-alloc + (B) per-slot
// `_ESBMC_arrcpy_2d(base->arr[i], ...)` at the outermost layer, and
// VSA admitted a null branch on reads of `base->arr[i]` inherited
// via the `*` (any-object) entry from the NONDET struct-tmp init in
// the clone's `cpp_new + *new_ptr = tmp` sequence.
//
// Root-cause fix (this commit):
//   1. value_set_domaint::transform now dispatches on ASSUME and calls
//      value_sett::apply_assume (previously a no-op default branch).
//   2. value_sett::apply_assume handles `p != 0` / `0 != p` guards by
//      stripping null-object / constant-zero entries from p's value-set.
//   3. _ESBMC_alloc_array / _ESBMC_alloc_array_sym add
//      `__ESBMC_assume(block != 0)` to declare their non-null return
//      contract — VSA now honours it so callers never see
//      `<0, 8, void>` in the points-to set for arr fields.
//   4. _ESBMC_arrcpy / _ESBMC_arrcpy_2d add `__ESBMC_assume(from_array
//      != 0)` to prune the residual `*` that propagates from upstream
//      NONDET struct inits.
//
// raw C / raw C++ equivalents at regression/esbmc-solidity/
// esol_clone_multi_dim_pass/repro_raw/ — they pass for a different
// reason (they implement their own `_arrcpy_2d` without the defensive
// null-check assertion), so they wouldn't have triggered the false
// positive even without the VSA fix. Kept for the
// GOTO-shape-is-not-the-problem conclusion that drove Phase A.
function __ESOL_deep_copy(C src) pure returns (C) { return src; }

contract C {
    uint256[2][2][3] arr;
    function setAt(uint256 i, uint256 j, uint256 k, uint256 v) public { arr[i][j][k] = v; }
    function get(uint256 i, uint256 j, uint256 k) public view returns (uint256) { return arr[i][j][k]; }
}

contract H {
    function check(uint256 a) public {
        require(a != 0);
        C base = new C();
        base.setAt(0, 0, 0, a);
        C clone = __ESOL_deep_copy(base);
        // Round-trip: clone should see base's write.
        assert(clone.get(0, 0, 0) == a);
    }
}
