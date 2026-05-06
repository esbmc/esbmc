// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Adversarial counterpart for encodeWithSelector: same dispatch via
// Logic.setX.selector, but the post-delegatecall assertion checks the
// wrong value.  Must FAIL to prove the selector form is actually
// routing through the shadow dispatch and propagating the argument.

contract Logic {
    uint256 public x;
    function setX(uint256 v) public { x = v; }
}

contract Proxy {
    uint256 public x;

    function test() public {
        x = 0;
        address(this).delegatecall(
            abi.encodeWithSelector(Logic.setX.selector, uint256(42))
        );
        assert(x == 999);
    }
}
