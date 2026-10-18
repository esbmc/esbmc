// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Isolation property: after clone, mutating base must NOT change clone.
// Tests the cloned instance is a genuinely separate storage slot for
// primitive fields.
function __ESOL_deep_copy(C src) pure returns (C) { return src; }

contract C {
    uint256 public u;
    function set(uint256 _u) public { u = _u; }
}

contract H {
    function check(uint256 a, uint256 b) public {
        C base = new C();
        base.set(a);
        C clone = __ESOL_deep_copy(base);
        // mutate base only
        base.set(b);
        // clone should still hold the snapshot
        assert(clone.u() == a);
    }
}
