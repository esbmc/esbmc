// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Regression for the 1-ether-payback bug detection at full ETH magnitude.
// `D`'s constructor sends 1 ether back to msg.sender. When `C` deploys
// `D` with `value: amount` from `check`, the wrapper sets msg.sender to
// C's $address inside D's ctor, so D's transfer credits C with 1 ether.
// Net balance change for C: -amount + 1 ether != -amount.
//
// Pre-fix history: this magnitude (1 ether == 10^18) never surfaced the
// bug because `(uint256_t)nondet_uint()` constrained msg_value/balance
// to [0, 2^32). Paths needing balance ≥ 1 ether were unsatisfiable, so
// the assertion was vacuously true and ESBMC reported SUCCESSFUL.
//
// Post-fix: nondet_uint256() gives true 256-bit nondet, the
// sufficient-balance path is feasible, and the assertion below holds
// because of the explicit `+ 1 ether` correction.
contract D {
    constructor() payable {
        payable(msg.sender).transfer(1 ether);
    }
}

contract C {
    function check(uint256 amount) external payable {
        uint256 pre = address(this).balance;
        D d = new D{value: amount}();
        uint256 post = address(this).balance;
        // Account for D's payback of 1 ether to C.
        assert(post == pre - amount + 1 ether);
    }
}
