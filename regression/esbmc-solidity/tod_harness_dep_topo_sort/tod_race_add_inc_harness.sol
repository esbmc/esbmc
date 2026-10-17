// Auto-generated TOD (Transaction Order Dependence) harness
// Contract: Base
// Pair:     add vs inc
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
function __ESOL_nondet_state_forward(Base c) {
    // replaced by ESBMC with a bounded nondet-dispatch loop
    // over c's public/external methods.
}
function __ESOL_deep_copy(Base src) pure returns (Base) {
    // replaced by ESBMC with _ESBMC_clone_Base: per-field deep copy of *src
    // into a fresh instance with a distinct $address and
    // independent heap-allocated array buffers.
    return src;
}

// ===== Dependencies =====
library L {
    function add(uint a, uint b) internal pure returns (uint) { return a + b; }
}

abstract contract IStep {
    function step() external virtual;
}

// ===== End dependencies =====

// ===== Target contract =====
contract Base is IStep {
    using L for uint;
    uint public x;

    function step() external override {
        x = x.add(1);
    }

    function add(uint v) public {
        x = x.add(v);
    }

    function inc() public {
        x = x.add(1);
    }
}

// ===== TOD harness =====
// ----- add vs inc -----
// Shared state variables (touched by both):
//   - x
contract TOD_add_inc {
    function test(
        uint256 a_v
    ) public {
        Base c1 = new Base();
        __ESOL_nondet_state_forward(c1);
        Base c2 = __ESOL_deep_copy(c1);
        require(address(c1) != address(c2), "isolate c1/c2");
        // Order 1: c1 runs add then inc
        c1.add(a_v);
        c1.inc();

        // Order 2: c2 runs inc then add
        c2.inc();
        c2.add(a_v);

        // Race check: if any shared state differs the pair is order-dependent
        __tod_race_check(c1.x() == c2.x());
    }
}

