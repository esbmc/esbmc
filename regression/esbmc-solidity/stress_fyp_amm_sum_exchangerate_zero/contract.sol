// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

// Regression: division-by-zero in `AutomatedMarketMakerSum.swapEthForTokens`
// from final-year-project-master/src/AutomatedMarketMakerSum.sol (line 120):
//
//     amountOut = (amountInMinusFee * 1e6) / exchangeRate[_propertyTokenAddress];
//
// `exchangeRate` is a `mapping(address => uint256)` with no initialiser.
// Before any call to `addLiquidity` runs `_updateExchangeRate` the slot is
// the default 0, so any `swapEthForTokens` issued between `createPool` and
// the first `addLiquidity` trips division by zero. The sibling functions
// `swapTokensForEth` (line 91) and `getEstimatedTokensForEth` (line 265) are
// on the same foot — the latter is even a `view` function, so a read-only
// caller can DoS.
//
// Defensive fix: require exchangeRate > 0 (or equivalently, require the pool
// has liquidity) before the division.
//
// As with the sibling regression (stress_fyp_amm_div_by_zero), ESBMC's
// Solidity frontend does not inline cross-contract external calls, so the
// attacker's literal call sequence (`new AMM; createPool; swapEthForTokens`)
// gets stubbed out. We collapse the sequence into a single constructor that
// reproduces the AMM state bit-for-bit at the moment of the bug and runs the
// exact arithmetic the AMM runs.

contract C {
    // Mirrors AutomatedMarketMakerSum.exchangeRate — unset defaults to 0.
    mapping(address => uint256) public exchangeRate;

    constructor() {
        // Attacker picks any address — here we use address(this) since we
        // have no external address available. The mapping is empty, so the
        // lookup returns the default 0. Identical to the state of
        // exchangeRate[_propertyTokenAddress] post-createPool.
        address pt = address(this);

        // Verbatim algebra from swapEthForTokens with msg.value = 1 (any
        // strictly-positive value tripping the same bug).
        uint256 value = 1;
        uint256 amountInMinusFee = (value * 990) / 1000;        // -> 0 (irrelevant)

        // DIVISION BY ZERO: exchangeRate[pt] == 0.
        uint256 amountOut = (amountInMinusFee * 1e6) / exchangeRate[pt];

        // prevent slicing
        exchangeRate[pt] = amountOut;
    }
}
