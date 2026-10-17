// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// The pointer-array bounds check also fires when the out-of-bounds access is in
// an `if` CONDITION (not a statement body): with an unconstrained k, `b[k]` on a
// length-2 array is out of bounds, so this must be caught as an array-bounds
// violation. Companion to the short-circuit-guarded no-false-positive test.
contract C {
    function f(uint k) public pure {
        uint[] memory b = new uint[](2);
        if (b[k] == 0) { assert(true); }   // OUT OF BOUNDS: k unconstrained
    }
}
