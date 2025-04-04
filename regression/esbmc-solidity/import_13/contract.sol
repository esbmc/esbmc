// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;
import "./contract2.sol";

contract _MAIN_ {
    Bank bank;

    uint a;

    constructor () {
        bank = new Bank();
        a = bank.a();
    }
    function test() public {
        a = a + c;

        assert(a == 4);
    }

}