// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.4;

// Tests custom error without arguments and plain revert().
contract AccessControl {
    address public owner;
    error Unauthorized();

    constructor() {
        owner = msg.sender;
    }

    function restrictedAction() public view {
        if (msg.sender != owner)
            revert Unauthorized();
        // Only owner reaches here
        assert(msg.sender == owner);
    }

    function plainRevert(bool fail) public pure {
        if (fail)
            revert();
        // If we reach here, fail was false
        assert(!fail);
    }
}
