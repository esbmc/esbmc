// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// A local dynamic memory array `new T[](n)` lowers to a *pointer*-backed
// buffer, so goto-check's native array-bounds check never fires and
// --no-standard-checks also disables the pointer-deref check. The explicit
// `pos < _ESBMC_array_length(a)` claim emitted by get_index_access_expr for
// pointer-backed dyn-arrays is what catches this: with a symbolic n the
// allocation may have length 0, so a[0] is out of bounds.
// Regression pin for the local pointer-backed array OOB fix (dynamic path).
contract MyContract {
    function run(uint8 n) public pure {
        uint8[] memory a = new uint8[](n);
        a[0] = 100;   // OUT OF BOUNDS when n == 0
    }
}
