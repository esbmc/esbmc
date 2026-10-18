// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Correct onlyOwner: only owner passes the require.
// In-function assert(msg.sender == owner) trivially holds because
// the require's ASSUME pins msg.sender == owner on every reached path.
contract Bug {
    address public owner;
    constructor() { owner = msg.sender; }
    modifier onlyOwner() { require(msg.sender == owner); _; }
    function privileged() public onlyOwner {
        assert(msg.sender == owner);
    }
}
