// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Fixed-size array isolation after __ESOL_deep_copy.
// Post-Phase-1 of the deep-copy walker, build_tod_clone_helper emits
// `c->arr = _ESBMC_arrcpy(base->arr, N, sizeof(uint256))`, giving
// clone its own heap slab.  A write to base.arr AFTER the clone
// MUST NOT be visible via clone.
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
        base.setAt(1, b); // mutate base AFTER clone
        // Isolation property: clone still sees the pre-clone value `a`,
        // NOT base's post-clone write `b`.
        assert(clone.get(1) == a);
    }
}
// KNOWNBUG: the inductive step cannot prove this without the inductive
// invariants the local k-induction inferred (loop-entry invariants,
// counted-loop skip, per-field havoc), which upstream's rewrite dropped.
