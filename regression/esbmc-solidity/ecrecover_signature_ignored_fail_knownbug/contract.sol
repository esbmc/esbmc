// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Pin-test (U3): ecrecover() ignores (v,r,s) — the model returns
// nondet address regardless of signature inputs (ledger #9). Two
// recovers with the same hash but different (v,r,s) should ideally
// give different addresses on real EVM. The current model returns
// independent nondet samples, so equality across calls is itself
// not stable. Pin a stronger invariant — recovery on the same
// hash should be deterministic — and document it as KNOWNBUG.
contract C {
    function test(bytes32 h, uint8 v1, bytes32 r1, bytes32 s1,
                  uint8 v2, bytes32 r2, bytes32 s2) public {
        address a1 = ecrecover(h, v1, r1, s1);
        address a2 = ecrecover(h, v2, r2, s2);
        // Under correct modelling, a1 should be a deterministic
        // function of (h, v, r, s), so same h with different
        // (v,r,s) might or might not give same a — but the SAME
        // call twice with the same args must give the same result.
        // Currently each call is fresh nondet, so even this weaker
        // form is broken — assert disjointness as KNOWNBUG.
        assert(a1 != a2 || (v1 == v2 && r1 == r2 && s1 == s2));
    }
}
