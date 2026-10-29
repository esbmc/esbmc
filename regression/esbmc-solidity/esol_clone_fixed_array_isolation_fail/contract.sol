// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Negative dual of esol_clone_fixed_array_isolation_pass.  Asserts
// the OPPOSITE outcome (clone sees base's post-clone write) — which
// is what would happen if the deep-copy walker in
// build_tod_clone_helper were reverted to a shallow pointer copy.
// Keeps the semantic firm: if anyone ever regresses the walker, this
// fail test flips to pass and the paired pass test flips to fail.
function __ESOL_deep_copy(C src) pure returns (C) { return src; }

contract C {
    uint256[3] public arr;
    function setAt(uint256 i, uint256 v) public { arr[i] = v; }
    function get(uint256 i) public view returns (uint256) { return arr[i]; }
}

contract H {
    function check(uint256 a, uint256 b) public {
        require(a != b);
        C base = new C();
        base.setAt(1, a);
        C clone = __ESOL_deep_copy(base);
        base.setAt(1, b);
        assert(clone.get(1) == b); // isolation => this FAILS
    }
}
