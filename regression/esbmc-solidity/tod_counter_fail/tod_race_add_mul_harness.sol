// Auto-generated TOD (Transaction Order Dependence) harness
// Contract: Counter
// Pair:     add vs mul
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

    function add(uint n) public {
        x = x + n;
    }

    function mul(uint n) public {
        x = x * n;
    }
}

// ===== TOD harness =====
// ----- add vs mul -----
// Shared state variables (touched by both):
//   - x
contract TOD_add_mul {
    function test(
        uint256 a_n,
        uint256 b_n
    ) public {
        Counter c1 = new Counter();
        __ESOL_nondet_state_forward(c1);
        Counter c2 = __ESOL_deep_copy(c1);
        require(address(c1) != address(c2), "isolate c1/c2");
        // Order 1: c1 runs add then mul
        c1.add(a_n);
        c1.mul(b_n);

        // Order 2: c2 runs mul then add
        c2.mul(b_n);
        c2.add(a_n);

        // Race check: if any shared state differs the pair is order-dependent
        __tod_race_check(c1.x() == c2.x());
    }
}

