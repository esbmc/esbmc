// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Regression: `.call{value:v}(data)` on a tracked-target address now
// returns `true` deterministically instead of a fresh `nondet_bool`.
// Previously `get_low_level_member_accsss` discarded the dispatch
// return and initialized the `(bool,bytes)` tuple's `success` slot
// to `nondet_bool`, so `(bool ok, ) = addr.call(...)` observed an
// arbitrary value even when the target's receive was known to
// complete.  Fix (src/solidity-frontend/solidity_convert_call.cpp):
// assign the dispatch's own return value into the tuple's `success`
// slot, so the $call#0/#1 body's guarantee (`true` on tracked-match,
// EOA-specific nondet/false on fallthrough) reaches the caller.
//
// `address(this)` under auto-dispatch resolves to
// `_ESBMC_Object_C.$address` — the same singleton the dispatch
// ladder compares against — so the self-call hits the tracked-
// match path.
contract C {
    uint256 public credited;
    receive() external payable { credited += msg.value; }

    function doCall() internal returns (bool) {
        (bool ok, ) = payable(address(this)).call{value: 1}("");
        return ok;
    }

    function test() public {
        require(address(this).balance >= 1);
        bool ok = doCall();
        assert(ok);
    }
}
