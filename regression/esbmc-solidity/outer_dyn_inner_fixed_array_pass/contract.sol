// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.8.0 <0.9.0;

// Round-trip test for `uint256[N][]` (outer dynamic, inner fixed).
// Under correct semantics the writes round-trip.
//
// Fixed 2026-04-24 by promoting the inner fixed-array slot to a native
// `array_typet(T, N)` inside the `is_dynarray_state` block of
// `solidity_convert_decl.cpp`. Previously the outer dynarray state-var
// promoted `pointer<pointer<T> + #sol_array_size>` to
// `array<pointer<T>, inf>` — leaving the inner slot pointer-sorted
// while push's zero-initializer and reads of unwritten slots produced
// ARRAY-sorted values. The sort divergence crashed cvc5/bitwuzla
// inside `array_convt::execute_array_ite`. Promoting the inner slot
// to `array_typet(T, N)` aligns both sides with a single ARRAY sort,
// eliminating the sort mismatch at the source.
//
// k-induction finds the inductive step at k=4 with 12 VCCs in under
// a second. Reads of unwritten slots (inductive step state) also
// work correctly — fresh valuations are ARRAY-sorted, matching the
// post-write sort.
contract OuterDynInnerFixedPass {
    uint256[3][] public grid;

    function check() external {
        grid.push();
        grid[0][0] = 10;
        grid[0][1] = 20;
        grid[0][2] = 30;
        assert(grid[0][0] == 10);
        assert(grid[0][1] == 20);
        assert(grid[0][2] == 30);
    }
}
