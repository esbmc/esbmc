// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// P3 regression-lock: pop on length=0 must revert (per Solidity spec:
// "Accessing an array past its end causes a failing assertion"; the
// library helper `_ESBMC_array_pop` already enforces `len > 0`).
//
// State-var dyn-array path (solidity_convert_ref.cpp:885) emits a bare
// `len-- ` without the check. 256-bit unsigned underflow makes
// `arr.length` become 2^256-1 — silent unsoundness; loops over
// `arr.length` walk forever.
//
// The assertion below uses 1_000_000 as a sanity cap. Pre-fix: post-pop
// length is 2^256-1, assertion fails. Post-fix (S3): the path becomes
// infeasible via __ESBMC_assume(len > 0); assertion holds vacuously.
//
// KNOWNBUG until S3 lands.
contract C {
    uint[] arr;

    function f() public {
        require(arr.length == 0);
        arr.pop();
        assert(arr.length < 1000000);
    }
}
