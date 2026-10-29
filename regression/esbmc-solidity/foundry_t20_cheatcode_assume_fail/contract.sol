// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
interface Vm { function assume(bool) external; }
abstract contract Test {
    Vm internal constant vm = Vm(0x7109709ECfa91a80626fF3989D68f67F5b1DD12D);
}
contract AssumeTest is Test {
    // assume constrains the fuzz input → assert holds
    function test_assume_pass(uint256 x) public {
        vm.assume(x < 10);
        assert(x < 100);
    }
    // assume does NOT over-constrain → x in [0,9], assert x<5 can fail
    function test_assume_fail(uint256 x) public {
        vm.assume(x < 10);
        assert(x < 5);
    }
}
