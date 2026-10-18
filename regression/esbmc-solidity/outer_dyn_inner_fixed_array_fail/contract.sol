// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.8.0 <0.9.0;

// Violation test for `uint256[N][]` (outer dynamic, inner fixed).
// Under correct semantics the `grid[0][0] != 10` assertion is violated.
//
// Fixed 2026-04-24 by the smt_conv select-decompose asymmetry patch:
// previously `convert_array_index` would decompose any 2-level index
// `a[i][j]` into a flat `i*N+j` whenever the *inner* slice was finite,
// even if the *outermost* array was infinite — the flat key was then
// applied against the outer infinite domain and the SMT encoder hit a
// nested array sort at a dangling pointer, crashing cvc5/bitwuzla
// during VCC encoding. The select check now mirrors the store check
// (tests the outermost array, not the immediate source).
//
// With a single write + single assertion the encoding now completes in
// under a second: VERIFICATION FAILED at k=1.
contract OuterDynInnerFixedFail {
    uint256[3][] public grid;

    function check() external {
        grid.push();
        grid[0][0] = 10;
        assert(grid[0][0] != 10);
    }
}
