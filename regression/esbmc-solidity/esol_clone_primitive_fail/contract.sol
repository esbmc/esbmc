// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Negative counterpart of esol_clone_primitive_pass: after the clone,
// mutate base independently and assert equality — must fail.
function __ESOL_deep_copy(C src) pure returns (C) { return src; }

contract C {
    uint256 public u;
    function set(uint256 _u) public { u = _u; }
}

contract H {
    function check(uint256 a, uint256 b) public {
        if (a == b) return;
        C base = new C();
        base.set(a);
        C clone = __ESOL_deep_copy(base);
        base.set(b);
        // Wrong: base diverged after clone, so clone.u() == base.u() iff a == b
        assert(clone.u() == base.u());
    }
}
