// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// FAIL dual of library_call_value_credit_pass.  Before C, library-
// scope `recipient.transfer(val)` was modeled as a pure nondet-bool
// return — no credit side, so EOA balance reads were ambiguous (a
// fresh nondet slab each look up).  After C, the library path runs
// `_ESBMC_eoa_credit(addr, val)` in its EOA fallthrough arm, so the
// recipient's balance slot is updated in place.  The assertion below
// claims the recipient's balance did NOT change after a library-
// wrapped transfer — refutable because the credit actually fires.
library Pay {
    function send(address payable recipient, uint256 val) internal {
        recipient.transfer(val);
    }
}

contract C {
    function test(address payable eoa) public {
        require(eoa != address(this));
        require(eoa != address(0));
        // Pin the EOA's starting balance so it is not fresh-nondet on
        // the second read.
        require(eoa.balance == 100);
        Pay.send(eoa, 10);
        // With C's fix, eoa.balance is now 110 (or more if the
        // over-approx nondet sender path touched it again).  Asserting
        // it stayed at 100 must FAIL.
        assert(eoa.balance == 100);
    }
}
