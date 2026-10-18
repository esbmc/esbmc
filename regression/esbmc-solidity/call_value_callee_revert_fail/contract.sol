// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// A low-level `.call{value:}` to a TRACKED target now models call failure:
// the dispatcher returns `ok = !reverted` instead of a hard-coded `true`,
// so a reverting receive/fallback is observable as `ok == false`.
//
// Here the self-call resolves to the tracked singleton _ESBMC_Object_C and
// hits the always-modelled tracked-match path.  When `reject == true` the
// receive reverts, so `ok` can be false and `assert(ok)` is refutable.
//
// Before the fix the success flag was a constant `true` and this assertion
// held spuriously (VERIFICATION SUCCESSFUL); the call-failure branch was
// never explored.  See call_value_callee_revert_pass for the dual.
contract C {
    bool reject;
    constructor(bool r) { reject = r; }

    receive() external payable {
        require(!reject, "rejected");
    }

    function test() public {
        require(address(this).balance >= 1);
        (bool ok, ) = payable(address(this)).call{value: 1}("");
        assert(ok);
    }
}
