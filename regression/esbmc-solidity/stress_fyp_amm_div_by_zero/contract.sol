// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

// Regression: division-by-zero in the `AutomatedMarketMaker.swapTokensForEth`
// pattern from final-year-project-master/src/AutomatedMarketMaker.sol.
//
// The vulnerable shape, lifted verbatim from the project (lines 88-90):
//
//     uint256 amountInMinusFee = (_amount * 990) / 1000;
//     amountOut = (pool.reserveEth * amountInMinusFee)
//                 / (pool.reservePropertyToken + amountInMinusFee);
//
// Attacker-controlled call sequence:
//   1. createPool(pt)                -> reserves are (0, 0)
//   2. swapTokensForEth(pt, 1)       -> amountInMinusFee = 990/1000 = 0
//                                       denominator      = 0 + 0     = 0
//
// The 1% fee rounding eats the only non-zero input, the denominator collapses,
// ESBMC's default division-by-zero check fires at symex time -> VERIFICATION
// FAILED. A defensive fix is `require(_amount * 990 / 1000 > 0)` plus a
// non-zero reserve precondition.
//
// ESBMC's Solidity frontend does not currently inline cross-contract external
// calls, so the exact project call graph (`new AMM(); amm.createPool(...);
// amm.swap(...)`) gets replaced with a nondet stub and the bug is hidden.
// To keep the regression deterministic we collapse the two-step sequence into
// a single contract whose constructor reproduces the exact state the AMM sees
// right after createPool (reserves = 0, 0) and then executes the exact swap
// algebra on _amount = 1. The math is bit-for-bit identical.

contract C {
    // Mirrors AutomatedMarketMaker.Pool state after createPool: both reserves 0.
    uint256 public reservePropertyToken;
    uint256 public reserveEth;

    constructor() {
        // reservePropertyToken and reserveEth default to 0 — this is exactly
        // the post-createPool state in the original contract.

        // Verbatim from AutomatedMarketMaker.swapTokensForEth with _amount = 1.
        uint256 _amount = 1;
        uint256 amountInMinusFee = (_amount * 990) / 1000;      // -> 0

        // DIVISION BY ZERO: (reservePropertyToken + amountInMinusFee) == 0.
        uint256 amountOut = (reserveEth * amountInMinusFee)
            / (reservePropertyToken + amountInMinusFee);

        // prevent the whole expression being sliced away
        reserveEth = amountOut;
    }
}
