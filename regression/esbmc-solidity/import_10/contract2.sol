// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;
import "./contract3.sol";

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