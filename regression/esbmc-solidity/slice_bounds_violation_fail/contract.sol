// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// FAIL companion to slice_bounds_pass: assert that the slice was reached
// AND s > e simultaneously. After ledger #14's fix, the slice's path-
// guard `__ESBMC_assume(s <= e)` makes this unsatisfiable as a counter-
// example — i.e., the assertion `s > e` is never violated because no
// path reaches it. But the assertion `s == e + 1` (a specific s > e)
// is also unreachable, so the model passes the assertion vacuously.
//
// To get a real FAILED, assert something that's true on EVERY feasible
// path: assert that the slice is reachable. The dispatcher will drive
// `check` with arbitrary s, e, so reachability is feasible iff s <= e.
// Asserting `s == e + 1 || true` is trivially true; instead, we assert
// `slice.length == 0`. After the slice produces a fresh nondet of any
// length, this assertion can fail (slice nondet length != 0).
contract S {
    function check(bytes calldata b, uint s, uint e) external pure {
        bytes calldata slice = b[s:e];
        // slice has a fresh nondet length; SMT can pick length != 0.
        assert(slice.length == 0);
    }
}
