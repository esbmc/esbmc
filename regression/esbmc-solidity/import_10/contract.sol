// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;
import "./contract2.sol";

contract Reproduction is Unknown {
    Mutex public mutex;

    // ESBMC_Object_mutex
    constructor(Mutex u) {
        mutex = u;
    }

    function setup() public {
        mutex.set(1);
    }

    function trigger() public {
        mutex.run();
    }

    function run() external override {
        mutex.set(2); // Modify x to trigger the assertion failure
    }
}
