// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
// Inherited fixed-size array: derived contract D inherits a
// `uint256[4] pots` state var from Base.  The clone walker sees a
// single flat merged struct (ESBMC inlines base-class fields), so
// the deep-copy fixup applies to the inherited array the same way
// as a directly-declared array.  Tests both post-clone equality and
// post-clone isolation (writes to base-derived pots visible only
// via base, not clone).
function __ESOL_deep_copy(D src) pure returns (D) { return src; }

contract Base {
    uint256[4] public pots;
    function putPot(uint256 i, uint256 v) public { pots[i] = v; }
    function getPot(uint256 i) public view returns (uint256) { return pots[i]; }
}

contract D is Base {
    uint256 public counter;
    function bump() public { counter += 1; }
}

contract H {
    function check(uint256 a, uint256 b) public {
        require(a != b);
        D base = new D();
        base.putPot(1, a);
        base.bump();
        D clone = __ESOL_deep_copy(base);
        // (a) post-clone equality via inherited getter
        assert(clone.getPot(1) == a);
        assert(clone.counter() == 1);
        // (b) isolation: mutating base's inherited array does not leak
        base.putPot(1, b);
        assert(clone.getPot(1) == a);
    }
}
// KNOWNBUG: the inductive step cannot prove this without the inductive
// invariants the local k-induction inferred (loop-entry invariants,
// counted-loop skip, per-field havoc), which upstream's rewrite dropped.
