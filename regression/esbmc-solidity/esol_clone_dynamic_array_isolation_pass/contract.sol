// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Dynamic array isolation — adversarial.  ESBMC models dynamic arrays
// via a global pool index; a naive bit-level struct copy would copy
// the index and leave clone+base sharing the same backing buffer,
// causing post-clone writes on base to be visible via clone.
//
// If deep_copy is implemented as a true per-instance snapshot, this
// test passes.  If it shares the buffer, the test fails — exposing a
// real soundness bug for TOD harnesses that touch dynamic arrays.
function __ESOL_deep_copy(C src) pure returns (C) { return src; }

contract C {
    uint256[] public arr;
    function push(uint256 v) public { arr.push(v); }
    function setAt(uint256 i, uint256 v) public { arr[i] = v; }
    function get(uint256 i) public view returns (uint256) { return arr[i]; }
}

contract H {
    function check(uint256 a, uint256 b) public {
        C base = new C();
        base.push(a);
        C clone = __ESOL_deep_copy(base);
        // mutate base only; clone[0] should remain `a`.
        base.setAt(0, b);
        assert(clone.get(0) == a);
    }
}
