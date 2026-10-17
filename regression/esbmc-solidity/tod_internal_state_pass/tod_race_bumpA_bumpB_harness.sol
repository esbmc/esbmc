// Auto-generated TOD (Transaction Order Dependence) harness
// Contract: C
// Pair:     bumpA vs bumpB
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
function __ESOL_nondet_state_forward(C c) {
    // replaced by ESBMC with a bounded nondet-dispatch loop
    // over c's public/external methods.
}
function __ESOL_deep_copy(C src) pure returns (C) {
    // replaced by ESBMC with _ESBMC_clone_C: per-field deep copy of *src
    // into a fresh instance with a distinct $address and
    // independent heap-allocated array buffers.
    return src;
}

// ===== Target contract =====
contract C {
    uint256 internal a;
    uint256 internal b;

    function bumpA(uint256 n) public { a = a + n; }
    function bumpB(uint256 n) public { b = b + n; }

    function __tod_get_a() public view returns (uint256) { return a; }
    function __tod_get_b() public view returns (uint256) { return b; }
}

// ===== TOD harness =====
// ----- bumpA vs bumpB -----
// Shared state variables (touched by both):
//   - a
//   - b
contract TOD_bumpA_bumpB {
    function test(
        uint256 a_n,
        uint256 b_n
    ) public {
        C c1 = new C();
        __ESOL_nondet_state_forward(c1);
        C c2 = __ESOL_deep_copy(c1);
        require(address(c1) != address(c2), "isolate c1/c2");
        // Order 1: c1 runs bumpA then bumpB
        c1.bumpA(a_n);
        c1.bumpB(b_n);

        // Order 2: c2 runs bumpB then bumpA
        c2.bumpB(b_n);
        c2.bumpA(a_n);

        // Race check: if any shared state differs the pair is order-dependent
        __tod_race_check(c1.__tod_get_a() == c2.__tod_get_a());
        __tod_race_check(c1.__tod_get_b() == c2.__tod_get_b());
    }
}

