// Auto-generated TOD (Transaction Order Dependence) harness
// Contract: MultiTod
// Pair:     addX vs mulX
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
// ----- addX vs mulX -----
// Shared state variables (touched by both):
//   - x
contract TOD_addX_mulX {
    function test(
        uint256 a_n,
        uint256 b_n
    ) public {
        MultiTod c1 = new MultiTod();
        __ESOL_nondet_state_forward(c1);
        MultiTod c2 = __ESOL_deep_copy(c1);
        require(address(c1) != address(c2), "isolate c1/c2");
        // Order 1: c1 runs addX then mulX
        c1.addX(a_n);
        c1.mulX(b_n);

        // Order 2: c2 runs mulX then addX
        c2.mulX(b_n);
        c2.addX(a_n);

        // Race check: if any shared state differs the pair is order-dependent
        __tod_race_check(c1.x() == c2.x());
    }
}

