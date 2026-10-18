// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Minimal forge-std-like stub (asset under test for the spike)
interface Vm {
    function warp(uint256) external;
    function prank(address) external;
    function deal(address, uint256) external;
    function assume(bool) external;
}

abstract contract Test {
    Vm internal constant vm = Vm(0x7109709ECfa91a80626fF3989D68f67F5b1DD12D);
}

contract Counter {
    uint256 public x;
    function inc() public { x += 1; }
}

contract CounterTest is Test {
    // A) does the harness run a test fn and does a WRONG native assert surface?
    function test_bug() public {
        Counter c = new Counter();
        c.inc();
        assert(c.x() == 2); // WRONG: x==1 → expect VERIFICATION FAILED if reached
    }

    // B) correct native assert
    function test_ok() public {
        Counter c = new Counter();
        c.inc();
        assert(c.x() == 1); // expect SUCCESSFUL
    }

    // C) does a vm.<cheatcode>() member call even parse/ingest? what does it lower to?
    function test_cheatcode() public {
        vm.warp(123);
        assert(block.timestamp == 123); // vm.warp unmodeled → timestamp nondet → expect FAILED
    }
}
