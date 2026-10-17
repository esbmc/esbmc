// Auto-generated TOD (Transaction Order Dependence) harness
// Contract: Counter
// Pair:     addA vs addB
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
    uint public counter;

    constructor() {
        counter = 0;
    }

    function addA() public {
        counter = counter + 1;
    }

    function addB() public {
        counter = counter + 1;
    }
}

// ===== TOD harness =====
// ----- addA vs addB -----
// Shared state variables (touched by both):
//   - counter
contract TOD_addA_addB {
    function test() public {
        Counter c1 = new Counter();
        __ESOL_nondet_state_forward(c1);
        Counter c2 = __ESOL_deep_copy(c1);
        require(address(c1) != address(c2), "isolate c1/c2");
        // Order 1: c1 runs addA then addB
        c1.addA();
        c1.addB();

        // Order 2: c2 runs addB then addA
        c2.addB();
        c2.addA();

        // Race check: if any shared state differs the pair is order-dependent
        __tod_race_check(c1.counter() == c2.counter());
    }
}

