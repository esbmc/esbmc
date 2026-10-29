// Auto-generated TOD (Transaction Order Dependence) harness
// Contract: Derived
// Pair:     takeSnapshot vs baseMul
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

// ===== Dependencies =====
contract Base {
    uint public baseX;

    constructor() {
        baseX = 0;
    }

    function baseAdd(uint n) public {
        baseX = baseX + n;
    }
}

// ===== End dependencies =====

// ===== Target contract =====
contract Derived is Base {
    uint public derivedY;

    constructor() {
        derivedY = 0;
    }

    // Reads baseX (inherited) and writes derivedY. baseMul() below
    // scales baseX, so swapping the two calls changes derivedY.
    function takeSnapshot() public {
        derivedY = baseX + derivedY;
    }

    function baseMul(uint n) public {
        baseX = baseX * n;
    }
}

// ===== TOD harness =====
// ----- takeSnapshot vs baseMul -----
// Shared state variables (touched by both):
//   - derivedY
contract TOD_takeSnapshot_baseMul {
    function test(
        Derived c1,
        Derived c2,
        uint256 b_n
    ) public {
        require(address(c1) != address(c2), "isolate c1/c2");
        // Order 1: c1 runs takeSnapshot then baseMul
        c1.takeSnapshot();
        c1.baseMul(b_n);

        // Order 2: c2 runs baseMul then takeSnapshot
        c2.baseMul(b_n);
        c2.takeSnapshot();

        // Race check: if any shared state differs the pair is order-dependent
        __tod_race_check(c1.derivedY() == c2.derivedY());
    }
}

