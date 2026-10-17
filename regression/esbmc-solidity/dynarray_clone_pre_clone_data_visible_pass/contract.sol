// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// T1.1 Stage S0 — clone preserves base's pre-clone data, even after
// base subsequently mutates.  The post-mutation read tests for true
// isolation: shared-SMT-array alias would let base.setAt(0, b) update
// clone's view; only a per-instance copy at clone time keeps clone's
// element [0] at the original `a`.
// Today: alias propagates base's setAt through to clone.
// Will flip to CORE at Stage S3 (T1.1) — clone walker copies elements.
function __ESOL_deep_copy(C src) pure returns (C) { return src; }

contract C {
    uint256[] public arr;
    function push(uint256 v) public { arr.push(v); }
    function setAt(uint256 i, uint256 v) public { arr[i] = v; }
    function get(uint256 i) public view returns (uint256) { return arr[i]; }
}

contract H {
    function check(uint256 a, uint256 b) public {
        if (a == b) return;
        C base = new C();
        base.push(a);
        C clone = __ESOL_deep_copy(base);
        base.setAt(0, b);  // base mutates after clone
        assert(clone.get(0) == a);  // clone must still see its original
    }
}
