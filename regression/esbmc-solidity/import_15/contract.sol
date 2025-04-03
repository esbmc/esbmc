// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

import "./contract2.sol";
import "./contract3.sol";

contract A {
    uint public totalValue;
    B public b;
    C public c;

    constructor() {
        b = new B();
        c = new C();
        totalValue = b.value() + c.value(); // 3 + 4 = 7
    }

    function test() public view {
        assert(totalValue == 7);
    }
}
