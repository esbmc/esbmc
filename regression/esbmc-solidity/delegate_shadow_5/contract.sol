// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Adversarial: Logic.earlyExit returns early via a conditional branch.
// Without the return-rewrite, the early return would escape the caller
// and skip the assertions. With the fix, the early return must only
// exit the inlined arm; Proxy.test() continues after the delegatecall.
//
// Case 1: v == 7 triggers the early branch that writes x = 100 and
//         returns without reaching x = v.  After the delegatecall,
//         Proxy.x must equal 100, and marker must equal 1.
// Case 2: any other v falls through to x = v.

contract Logic {
    uint256 public x;
    function earlyExit(uint256 v) public returns (uint256) {
        if (v == 7) {
            x = 100;
            return 999;
        }
        x = v;
        return v;
    }
}

contract Proxy {
    uint256 public x;
    uint256 public marker;

    function test() public {
        x = 0;
        marker = 0;
        address(this).delegatecall(
            abi.encodeWithSignature("earlyExit(uint256)", uint256(7))
        );
        marker = 1;
        assert(x == 100);
        assert(marker == 1);
    }
}
