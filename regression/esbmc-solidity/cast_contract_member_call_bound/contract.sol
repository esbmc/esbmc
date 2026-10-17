// SPDX-License-Identifier: MIT
// Regression for the address→contract cast member-call fix, bound mode.
//
// Pre-fix: `ICallback(cb).onCall(v)` tripped the frontend with
//   ERROR: Expecting contract type
//   unsignedbv width: 160 #sol_type: ADDRESS
//   ERROR: CONVERSION ERROR
// because the cast's result kept its underlying `address` type instead
// of being re-typed as CONTRACT before member-access dispatch.
//
// Post-fix: cast is re-wrapped as a pointer to the target contract's
// singleton (with `$address` carrying the cast's address value), so
// dispatch proceeds into ICallback's implementation body under --bound.
//
// The assertion below holds regardless of the callback's behaviour
// (cb's state is separate from ours), so verification must succeed.
pragma solidity >=0.8.0;

interface ICallback {
    function onCall(uint256 v) external;
}

contract Callback is ICallback {
    uint256 public last;
    function onCall(uint256 v) external override { last = v; }
}

contract Caller {
    uint256 public marker = 111;

    function doCall(address cb, uint256 v) public {
        ICallback(cb).onCall(v);  // the pattern under test
    }

    function check(address cb, uint256 v) public {
        doCall(cb, v);
        // Caller's own state untouched by the external call.
        assert(marker == 111);
    }
}
