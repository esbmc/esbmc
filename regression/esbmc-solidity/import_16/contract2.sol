// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;
import "./contract.sol";

contract DerivedContract is BaseContract {
    constructor() {

        assert(value != 42);
    }
    
}