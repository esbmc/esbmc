// Auto-generated TOD (Transaction Order Dependence) harness
// Contract: WinToken
// Pair:     addPot vs claimWin
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
function __ESOL_nondet_state_forward(WinToken c) {
    // replaced by ESBMC with a bounded nondet-dispatch loop
    // over c's public/external methods.
}
function __ESOL_deep_copy(WinToken src) pure returns (WinToken) {
    // replaced by ESBMC with _ESBMC_clone_WinToken: per-field deep copy of *src
    // into a fresh instance with a distinct $address and
    // independent heap-allocated array buffers.
    return src;
}

// ===== Target contract =====
contract WinToken {
    uint256 public pot;
    address public winner;

    function addPot(uint256 amount) public {
        pot = pot + amount;
    }

    function claimWin(uint256 threshold) public {
        if (pot >= threshold) {
            winner = msg.sender;
        }
    }
}

// ===== TOD harness =====
// ----- addPot vs claimWin -----
// Shared state variables (touched by both):
//   - pot
//   - winner
contract TOD_addPot_claimWin {
    function test(
        uint256 a_amount,
        uint256 b_threshold
    ) public {
        WinToken c1 = new WinToken();
        __ESOL_nondet_state_forward(c1);
        WinToken c2 = __ESOL_deep_copy(c1);
        require(address(c1) != address(c2), "isolate c1/c2");
        // Order 1: c1 runs addPot then claimWin
        c1.addPot(a_amount);
        c1.claimWin(b_threshold);

        // Order 2: c2 runs claimWin then addPot
        c2.claimWin(b_threshold);
        c2.addPot(a_amount);

        // Race check: if any shared state differs the pair is order-dependent
        __tod_race_check(c1.pot() == c2.pot());
        __tod_race_check(c1.winner() == c2.winner());
    }
}

