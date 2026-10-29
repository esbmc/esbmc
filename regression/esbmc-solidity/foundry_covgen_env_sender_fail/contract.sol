// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
// Inline msg.sender == owner gate (no modifier): the deployer-coordination
// must pin vm.startPrank(owner) in setUp and vm.prank per call so BOTH the
// owner arm and the non-owner arm are reachable in forge.
contract OwnerGate {
    address owner;
    uint256 public v;
    constructor() { owner = msg.sender; }
    function act() external {
        if (msg.sender == owner) v = 1;
        else v = 2;
    }
}
