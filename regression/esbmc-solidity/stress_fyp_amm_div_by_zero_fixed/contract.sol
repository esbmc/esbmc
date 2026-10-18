// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

// Companion "pass" case for stress_fyp_amm_div_by_zero.
//
// Same call shape as AutomatedMarketMaker.swapTokensForEth, but with the
// minimal defensive fix: require the fee-adjusted input to be non-zero. That
// precludes the (0, 0) denominator state reachable from
// createPool → swapTokensForEth(_amount = 1), so ESBMC's division-by-zero
// check no longer fires and verification succeeds.

contract C {
    uint256 public reservePropertyToken;
    uint256 public reserveEth;

    constructor() {
        uint256 _amount = 1000;  // attacker can no longer pick _amount = 1
        uint256 amountInMinusFee = (_amount * 990) / 1000;

        // Fix: ensure the denominator can never be zero.
        require(amountInMinusFee > 0);

        uint256 amountOut = (reserveEth * amountInMinusFee)
            / (reservePropertyToken + amountInMinusFee);
        reserveEth = amountOut;
    }
}
