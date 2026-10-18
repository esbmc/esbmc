// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

// Companion "pass" case for stress_fyp_amm_sum_exchangerate_zero.
//
// Same shape as AutomatedMarketMakerSum.swapEthForTokens but guarded against
// the uninitialised-exchangeRate case. Once the guard holds, the division
// cannot trip and verification succeeds.

contract C {
    mapping(address => uint256) public exchangeRate;

    constructor() {
        address pt = address(this);
        // Seed a non-zero exchange rate as addLiquidity would have.
        exchangeRate[pt] = 1e6;

        uint256 value = 1;
        uint256 amountInMinusFee = (value * 990) / 1000;

        // Fix: refuse to divide when the pool has no rate yet.
        require(exchangeRate[pt] > 0);

        uint256 amountOut = (amountInMinusFee * 1e6) / exchangeRate[pt];
        exchangeRate[pt] = amountOut + 1;
    }
}
