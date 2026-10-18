// Auto-generated TOD (Transaction Order Dependence) harness
// Contract: C
// Pair:     setR vs claimR
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
    uint256 internal r;
    bool private claimed;

    function setR(uint256 v) public {
        require(!claimed);
        r = v;
    }

    function claimR() public {
        require(!claimed);
        claimed = true;
    }

    function __tod_get_r() public view returns (uint256) { return r; }
    function __tod_get_claimed() public view returns (bool) { return claimed; }
}

// ===== TOD harness =====
// ----- setR vs claimR -----
// Shared state variables (touched by both):
//   - r
//   - claimed
contract TOD_setR_claimR {
    function test(
        uint256 a_v
    ) public {
        C c1 = new C();
        __ESOL_nondet_state_forward(c1);
        C c2 = __ESOL_deep_copy(c1);
        require(address(c1) != address(c2), "isolate c1/c2");
        // Order 1: c1 runs setR then claimR
        try c1.setR(a_v) {} catch {}
        try c1.claimR() {} catch {}

        // Order 2: c2 runs claimR then setR
        try c2.claimR() {} catch {}
        try c2.setR(a_v) {} catch {}

        // Race check: if any shared state differs the pair is order-dependent
        __tod_race_check(c1.__tod_get_r() == c2.__tod_get_r());
        __tod_race_check(c1.__tod_get_claimed() == c2.__tod_get_claimed());
    }
}

