// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;
import "./contract3.sol";

contract Derived is Base {
    constructor() Base() {
        data = 2;
    }
    // The overriding function may only change the visibility of the overridden function from external to public
}