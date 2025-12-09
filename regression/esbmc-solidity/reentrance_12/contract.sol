// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

interface Unknown {
    function run() external;
}

contract Mutex {
    uint x;
    bool lock;

    Unknown immutable unknown;

    constructor(Unknown u) {
        require(address(u) != address(0));
        unknown = u;
    }

    function set(uint x_) public {
        require(!lock);
        lock = true;
        x = x_;
        lock = false;
    }

    function run() public {
        require(!lock);
        lock = true;
        uint xPre = x;
        unknown.run();
        assert(xPre == x);
        lock = false;
    }
}
