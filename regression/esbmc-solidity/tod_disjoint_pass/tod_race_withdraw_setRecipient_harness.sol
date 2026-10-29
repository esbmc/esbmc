// Auto-generated TOD (Transaction Order Dependence) harness
// Contract: Fund
// Pair:     withdraw vs setRecipient
// Mode:     race

// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// TOD classification helpers.  An assertion failure inside one
// of these functions tells the user which TOD category fired.
function __tod_race_check(bool cond) pure {
    assert(cond); // TOD-Race: non-commutative state update
}
function __tod_balance_check(bool cond) pure {
    assert(cond); // TOD-Balance: order-dependent ETH movement
}

// ESBMC intrinsic stubs (the frontend ignores the bodies).
function __ESOL_nondet_state_forward(Fund c) {
    // replaced by ESBMC with a bounded nondet-dispatch loop
    // over c's public/external methods.
}
function __ESOL_deep_copy(Fund src) pure returns (Fund) {
    // replaced by ESBMC with _ESBMC_clone_Fund: per-field deep copy of *src
    // into a fresh instance with a distinct $address and
    // independent heap-allocated array buffers.
    return src;
}

// ===== Target contract =====
contract Fund {
    address public recipient;
    uint public balance;

    constructor() {
        balance = 100;
        recipient = address(0x1);
    }

    function withdraw() public {
        balance = 0;
    }

    function setRecipient(address newRecipient) public {
        recipient = newRecipient;
    }
}

// ===== TOD harness =====
// ----- withdraw vs setRecipient -----
// Shared state variables (touched by both):
//   - recipient
//   - balance
contract TOD_withdraw_setRecipient {
    function test(
        address b_newRecipient
    ) public {
        Fund c1 = new Fund();
        __ESOL_nondet_state_forward(c1);
        Fund c2 = __ESOL_deep_copy(c1);
        require(address(c1) != address(c2), "isolate c1/c2");
        // Order 1: c1 runs withdraw then setRecipient
        c1.withdraw();
        c1.setRecipient(b_newRecipient);

        // Order 2: c2 runs setRecipient then withdraw
        c2.setRecipient(b_newRecipient);
        c2.withdraw();

        // Race check: if any shared state differs the pair is order-dependent
        __tod_race_check(c1.recipient() == c2.recipient());
        __tod_race_check(c1.balance() == c2.balance());
    }
}

