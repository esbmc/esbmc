// Auto-generated TOD (Transaction Order Dependence) harness
// Contract: MultiTod
// Pair:     doubleY vs setY
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
function __ESOL_nondet_state_forward(MultiTod c) {
    // replaced by ESBMC with a bounded nondet-dispatch loop
    // over c's public/external methods.
}
function __ESOL_deep_copy(MultiTod src) pure returns (MultiTod) {
    // replaced by ESBMC with _ESBMC_clone_MultiTod: per-field deep copy of *src
    // into a fresh instance with a distinct $address and
    // independent heap-allocated array buffers.
    return src;
}

// ===== Target contract =====
contract MultiTod {
    uint public x;
    uint public y;

    constructor() { x = 1; y = 1; }

    function addX(uint n) public { x = x + n; }
    function mulX(uint n) public { x = x * n; }

    function setY(uint v) public { y = v; }
    function doubleY() public { y = y * 2; }
}

// ===== TOD harness =====
// ----- doubleY vs setY -----
// Shared state variables (touched by both):
//   - y
contract TOD_doubleY_setY {
    function test(
        uint256 b_v
    ) public {
        MultiTod c1 = new MultiTod();
        __ESOL_nondet_state_forward(c1);
        MultiTod c2 = __ESOL_deep_copy(c1);
        require(address(c1) != address(c2), "isolate c1/c2");
        // Order 1: c1 runs doubleY then setY
        c1.doubleY();
        c1.setY(b_v);

        // Order 2: c2 runs setY then doubleY
        c2.setY(b_v);
        c2.doubleY();

        // Race check: if any shared state differs the pair is order-dependent
        __tod_race_check(c1.y() == c2.y());
    }
}

