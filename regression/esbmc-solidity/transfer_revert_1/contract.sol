// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Bound-mode transfer balance accounting:
// When the caller has sufficient balance, transfer(val) must decrement
// the caller's own balance by exactly val. This demonstrates that the
// bound-mode helper actually performs the balance update on the caller
// (this) rather than being a no-op.
contract Receiver {
    receive() external payable {}
}

contract TransferBalance {
    Receiver r;

    function __ESBMC_assume(bool) internal pure {}

    constructor() {
        r = new Receiver();
    }

    function test() public {
        __ESBMC_assume(address(this).balance >= 100);
        uint256 b0 = address(this).balance;
        payable(address(r)).transfer(40);
        // Caller is the static TransferBalance instance under --contract
        // mode, so balance updates on `this` are observable.
        assert(address(this).balance == b0 - 40);
    }
}
