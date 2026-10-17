// Auto-generated TOD (Transaction Order Dependence) harness
// Contract: Counter
// Pair:     set5 vs set10
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
function __ESOL_nondet_state_forward(Counter c) {
    // replaced by ESBMC with a bounded nondet-dispatch loop
    // over c's public/external methods.
}
function __ESOL_deep_copy(Counter src) pure returns (Counter) {
    // replaced by ESBMC with _ESBMC_clone_Counter: per-field deep copy of *src
    // into a fresh instance with a distinct $address and
    // independent heap-allocated array buffers.
    return src;
}

// ===== Target contract =====
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

// ===== TOD harness =====
// ----- set5 vs set10 -----
// Shared state variables (touched by both):
//   - x
contract TOD_set5_set10 {
    function test() public {
        Counter c1 = new Counter();
        __ESOL_nondet_state_forward(c1);
        Counter c2 = __ESOL_deep_copy(c1);
        require(address(c1) != address(c2), "isolate c1/c2");
        // Order 1: c1 runs set5 then set10
        try c1.set5() {} catch {}
        try c1.set10() {} catch {}

        // Order 2: c2 runs set10 then set5
        try c2.set10() {} catch {}
        try c2.set5() {} catch {}

        // Race check: if any shared state differs the pair is order-dependent
        __tod_race_check(c1.x() == c2.x());
    }
}

