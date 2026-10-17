// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
interface Vm { function expectRevert() external; }
abstract contract Test { Vm internal constant vm = Vm(0x7109709ECfa91a80626fF3989D68f67F5b1DD12D); }
contract Bank {
    function withdraw(uint256 amt) external pure {
        require(amt <= 100, "too much");
    }
}
contract BankTest is Test {
    // correct: expects revert, and the call DOES revert (amt>100) → SUCCESSFUL
    function test_revert_ok() public {
        Bank b = new Bank();
        vm.expectRevert();
        b.withdraw(200);
    }
    // wrong: expects revert, but the call does NOT revert (amt<=100) → FAILED
    function test_revert_wrong() public {
        Bank b = new Bank();
        vm.expectRevert();
        b.withdraw(50);
    }
}
