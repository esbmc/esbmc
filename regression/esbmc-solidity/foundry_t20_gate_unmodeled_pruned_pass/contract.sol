// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
interface Vm { function deal(address,uint256) external; }
abstract contract Test { Vm internal constant vm = Vm(0x7109709ECfa91a80626fF3989D68f67F5b1DD12D); }
contract DealTest is Test {
    function test_deal_unmodeled() public {
        vm.deal(address(this), 1);   // UNMODELED -> prune mode kills the path here
        assert(1 == 2);              // never checked (pruned) -> SUCCESSFUL (prune's vacuous posture)
    }
}
