// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;
import "./contract4.sol";
import "./contract2.sol";

contract DD is Base, Derived {
    constructor() {
        assert(data == 2);
    }
    function test1() public view {
        data;
        assert(data == 2);
    }
}