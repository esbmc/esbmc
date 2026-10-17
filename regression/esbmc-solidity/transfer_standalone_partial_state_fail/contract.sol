// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Ledger #16, FAIL companion: assert that x == 1 is observable. This is the
// stuck-intermediate-state assertion. It SHOULD fail because no path leaves
// x at the intermediate value (real EVM rolls back on failure; success
// overwrites to 2). Model agrees because the failure branch is pruned by
// assume(false), so paths only reach `x == 1` transiently inside pay().
contract Vault {
    uint public x;
    function pay(address payable to, uint amt) public {
        x = 1;
        to.transfer(amt);
        x = 2;
    }
    receive() external payable {}
}

contract H {
    Vault a;
    Vault b;
    constructor() {
        a = new Vault();
        b = new Vault();
    }
    function trigger(uint amt) public {
        a.pay(payable(address(b)), amt);
    }
    function check_x_stuck_intermediate() public view {
        // sibling iter reading a.x(): cannot ever observe 1.
        assert(a.x() == 1);
    }
}
