// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// delegate shadow with return value: Logic.bump returns the new x.
// The return value must NOT cause the caller function to exit early.
// After the delegatecall, test() must continue executing, check that
// x was actually written to (proving the body ran), and also verify
// its OWN post-delegatecall logic runs (no early return escape).

contract Logic {
    uint256 public x;
    function bump(uint256 v) public returns (uint256) {
        x = v;
        return v + 1;
    }
}

contract Proxy {
    uint256 public x;
    uint256 public marker;

    function test() public {
        x = 0;
        marker = 0;
        address(this).delegatecall(
            abi.encodeWithSignature("bump(uint256)", uint256(42))
        );
        // These two assignments MUST run. If the inlined `return v + 1;`
        // escapes the caller, marker stays 0 and the assertion fires.
        marker = 1;
        assert(x == 42);
        assert(marker == 1);
    }
}
