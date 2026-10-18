// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// OOB on a NON-symbol dynamic-array base: `make(n)[5]` indexes the array
// returned by a call, which lowers to a malloc'd region. The frontend explicit
// array-bounds assert deliberately skips non-symbol bases (to avoid evaluating
// a side-effecting base twice), so this OOB surfaces ONLY as a *dereference*
// failure — detectable only because --bounds-check re-enables the pointer-deref
// check (esbmc_parseoptions.cpp couples no-pointer-check to bounds-check).
// Regression pin for that coupling: without it this reads VERIFICATION UNKNOWN.
contract C {
    function make(uint n) internal pure returns (uint[] memory a) { a = new uint[](n); }
    function f(uint n) public pure returns (uint) {
        return make(n)[5];   // OUT OF BOUNDS when n <= 5
    }
}
