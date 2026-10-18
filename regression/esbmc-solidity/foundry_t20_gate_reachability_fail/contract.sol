// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
interface Vm { function deal(address,uint256) external; }
abstract contract Test { Vm internal constant vm = Vm(0x7109709ECfa91a80626fF3989D68f67F5b1DD12D); }
contract ReachTest is Test {
    function test_bug_before_unmodeled() public {
        assert(1 == 2);              // real bug BEFORE the unmodeled cheatcode
        vm.deal(address(this), 1);   // prune only affects AFTER here
    }
}
