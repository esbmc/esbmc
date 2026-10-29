pragma solidity ^0.8.0;

contract MyContract {
    struct T {
        uint[4] x;
    }
    function callVuln() public {
        uint[4] memory xx = [uint(1), 2, 3, 4];
        T memory tInstance = T(xx);
        assert(tInstance.x[0] == 1); // fail
    }
}
// KNOWNBUG: the inductive step cannot prove this without the inductive
// invariants the local k-induction inferred (loop-entry invariants,
// counted-loop skip, per-field havoc), which upstream's rewrite dropped.
