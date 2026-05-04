// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Adversarial counterpart to delegate_shadow_7: same helper-inlining
// scenario with swapped Proxy layout, but with deliberately wrong
// post-delegatecall assertions. Must FAIL — proves the helper's state
// writes are actually visible and checked on the Proxy side.

contract Logic {
    uint256 public x;
    uint256 public y;

    function _setY(uint256 v) internal {
        y = v;
    }

    function setBoth(uint256 a, uint256 b) public {
        x = a;
        _setY(b);
    }
}

contract Proxy {
    uint256 public y;
    uint256 public x;

    function test() public {
        x = 0;
        y = 0;
        address(this).delegatecall(
            abi.encodeWithSignature("setBoth(uint256,uint256)", uint256(10), uint256(20))
        );
        // Expected: x == 10, y == 20. This assert says y == 999 which
        // must FAIL once the helper actually writes through Proxy.y.
        assert(y == 999);
    }
}
