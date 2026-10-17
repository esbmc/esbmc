// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// SWC-115 fixed variant: the guard uses `msg.sender == owner`, which
// is the correct authorisation predicate.  No relay chain can satisfy
// it without `msg.sender` actually being `owner`, so the oracle
// `assert(msg.sender == owner)` always holds.
contract Vault {
    address public owner;
    uint256 public secret;

    constructor() {
        owner = msg.sender;
    }

    function setSecret(uint256 v) external {
        require(msg.sender == owner);
        assert(msg.sender == owner);
        secret = v;
    }
}
