// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// PASS dual of tod_internal_state_fail: `bumpA` and `bumpB` each write
// a distinct internal state variable, so there is no TOD race between
// them — the assertion over the UNION of writes still passes because
// both functions commute on disjoint storage.
//
// Exercises the shadow-getter path: `a` and `b` are internal, no
// Solidity auto-getter exists, but the harness can still diff the
// post-call state via the injected `__tod_get_a` / `__tod_get_b`.
contract C {
    uint256 internal a;
    uint256 internal b;

    function bumpA(uint256 n) public { a = a + n; }
    function bumpB(uint256 n) public { b = b + n; }
}
