// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// B5-B Stage S0 — negative baseline. Same shape as
// cross_iter_internal_revert_leak_pass, but the internal helper does
// NOT revert. State writes from earlier iterations propagate normally.
// `x` can be 0 (if check runs before any tryWrite) or 1 (after).
// Assertion is loose enough to hold in either order — guards against
// regressions in the harness wiring.
contract H {
    uint256 public x;

    function _helper() internal {
        x = 1;
    }

    function tryWrite() external {
        _helper();
    }

    function check() external view {
        assert(x == 0 || x == 1);
    }
}
