// Auto-generated TOD (Transaction Order Dependence) harness
// Contract: Leaf
// Pair:     bumpV vs setV
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
function __ESOL_nondet_state_forward(Leaf c) {
    // replaced by ESBMC with a bounded nondet-dispatch loop
    // over c's public/external methods.
}
function __ESOL_deep_copy(Leaf src) pure returns (Leaf) {
    // replaced by ESBMC with _ESBMC_clone_Leaf: per-field deep copy of *src
    // into a fresh instance with a distinct $address and
    // independent heap-allocated array buffers.
    return src;
}

// ===== Dependencies =====
contract Base {
    uint256 internal v;
    function setV(uint256 n) public { v = n; }
}

// ===== End dependencies =====

// ===== Target contract =====
contract Leaf is Base {
    function bumpV(uint256 n) public { v = v + n; }

    function __tod_get_v() public view returns (uint256) { return v; }
}

// ===== TOD harness =====
// ----- bumpV vs setV -----
// Shared state variables (touched by both):
//   - v
contract TOD_bumpV_setV {
    function test(
        uint256 a_n,
        uint256 b_n
    ) public {
        Leaf c1 = new Leaf();
        __ESOL_nondet_state_forward(c1);
        Leaf c2 = __ESOL_deep_copy(c1);
        require(address(c1) != address(c2), "isolate c1/c2");
        // Order 1: c1 runs bumpV then setV
        c1.bumpV(a_n);
        c1.setV(b_n);

        // Order 2: c2 runs setV then bumpV
        c2.setV(b_n);
        c2.bumpV(a_n);

        // Race check: if any shared state differs the pair is order-dependent
        __tod_race_check(c1.__tod_get_v() == c2.__tod_get_v());
    }
}

