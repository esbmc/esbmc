// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Regression for harness try/catch wrap on require-guarded callees.
//
// set5 and set10 race on `x`; whichever is first commits and the second
// reverts because its `require(x == 0)` fails.  In EVM these are two
// independent transactions — whoever is mined first wins and the other
// is discarded without affecting the winner's commit.  The two orderings
// therefore end at x=5 vs x=10 respectively — a real TOD.
//
// Pre-fix: the harness emitted `c1.set5(); c1.set10();` as two sequential
// external calls in the same `test()` harness function.  The second
// call's `require` fail emits `__ESBMC_assume(false)` which prunes the
// enclosing path, and since the same happens for the Order 2 path, every
// ordering gets pruned before the equality assertions are reached.
// Result: vacuous VERIFICATION SUCCESSFUL — the TOD is MISSED.
//
// Post-fix: the harness detects that set5/set10 contain an explicit
// `require`, so it wraps each call as `try c.set5() {} catch {}`.  The
// catch arm gives ESBMC an alternate path where the reverting call is
// skipped, leaving `test()` alive to reach the equality assertions.
// ESBMC then finds a feasible counter-example (c1.x=5 vs c2.x=10) and
// reports VERIFICATION FAILED — the TOD is detected.
contract Counter {
    uint public x;

    constructor() {
        x = 0;
    }

    function set5() public {
        require(x == 0);
        x = 5;
    }

    function set10() public {
        require(x == 0);
        x = 10;
    }
}
