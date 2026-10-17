// SPDX-License-Identifier: MIT
// Regression for the address→contract cast member-call fix, unbound mode.
//
// Pre-fix: same "Expecting contract type" error as the bound variant —
// the bug is in the frontend's TypeConversion → MemberAccess pipeline,
// which runs in both --bound and --unbound.
//
// Post-fix: under --unbound, the external call routes through
// `_ESBMC_Nondet_Extcall_ICallback()` (nondet havoc) instead of
// executing the callback body.  The caller's own state remains
// unchanged — unbound mode does not model cross-contract state
// mutation — so the assertion still holds.
pragma solidity >=0.8.0;

interface ICallback {
    function onCall(uint256 v) external;
}

contract Callback is ICallback {
    uint256 public last;
    function onCall(uint256 v) external override { last = v; }
}

contract Caller {
    uint256 public marker = 222;

    function doCall(address cb, uint256 v) public {
        ICallback(cb).onCall(v);  // the pattern under test
    }

    function check(address cb, uint256 v) public {
        doCall(cb, v);
        assert(marker == 222);
    }
}
