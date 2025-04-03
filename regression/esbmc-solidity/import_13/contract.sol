// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;
import "./contract2.sol";

contract _MAIN_ {
    Bank bank;

    constructor () public {
        bank = new Bank();
        uint a = 0;
        a = bank.a();

        assert(a == 3);
        assert(a == 4);
    }
}