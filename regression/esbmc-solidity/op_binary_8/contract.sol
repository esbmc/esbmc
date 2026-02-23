// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

contract powtest {
    uint256 public rewardedAmount;
    uint256 public frozenAmount;
    uint8 public decimals = 8;

    constructor() public {
        rewardedAmount = 300 * 10 ** uint256(decimals);
        assert(rewardedAmount == 30000000000);
    }

    function pow() public view{
        uint256 a = 10;
        assert(a ** decimals == 100000000);
    }


}
