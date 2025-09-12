// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

interface Unknown {
    function run() external;
}

contract Mutex {
    uint x;
    Unknown unknown;
    constructor(address _addr) {
        unknown = Unknown(_addr);
    }

    function set(uint x_) public {
        x = x_;
    }

    function run() public {
        uint xPre = x;
        unknown.run();
        assert(xPre == x);
    }
}

// SMTChecker: External Calls and Reentrancy

contract Reproduction is Unknown {
    Mutex public mutex;

    // ESBMC_Object_mutex
    constructor(address _addr) {
        mutex = Mutex(_addr);
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
