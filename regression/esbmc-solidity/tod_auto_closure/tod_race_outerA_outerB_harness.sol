// Auto-generated TOD (Transaction Order Dependence) harness
// Contract: Closure
// Pair:     outerA vs outerB
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
function __ESOL_nondet_state_forward(Closure c) {
    // replaced by ESBMC with a bounded nondet-dispatch loop
    // over c's public/external methods.
}
function __ESOL_deep_copy(Closure src) pure returns (Closure) {
    // replaced by ESBMC with _ESBMC_clone_Closure: per-field deep copy of *src
    // into a fresh instance with a distinct $address and
    // independent heap-allocated array buffers.
    return src;
}

// ===== Target contract =====
contract Closure {
    uint public x;

    constructor() { x = 0; }

    function _doubleX() internal { x = x * 2; }
    function _addOne() internal { x = x + 1; }

    function outerA() public { _doubleX(); }
    function outerB() public { _addOne(); }
}

// ===== TOD harness =====
// ----- outerA vs outerB -----
// Shared state variables (touched by both):
//   - x
contract TOD_outerA_outerB {
    function test() public {
        Closure c1 = new Closure();
        __ESOL_nondet_state_forward(c1);
        Closure c2 = __ESOL_deep_copy(c1);
        require(address(c1) != address(c2), "isolate c1/c2");
        // Order 1: c1 runs outerA then outerB
        c1.outerA();
        c1.outerB();

        // Order 2: c2 runs outerB then outerA
        c2.outerB();
        c2.outerA();

        // Race check: if any shared state differs the pair is order-dependent
        __tod_race_check(c1.x() == c2.x());
    }
}

