// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "./contract4.sol";
import "./contract5.sol";

contract B {
    uint public value;
    D public d;
    E public e;

    constructor() {
        d = new D();
        e = new E();
        value = d.value() + e.value(); // 1 + 2 = 3
    }
}