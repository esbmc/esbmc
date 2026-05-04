// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Internal helper inlining: Logic.setBoth calls a private helper _setY
// that writes a state variable. Proxy declares its fields in a DIFFERENT
// order than Logic. The old v1/v2 path emitted `_setY((Logic*)this, ...)`
// which accessed y at Logic's struct offset — which is Proxy's x slot
// because the layouts differ. The v3 path inlines _setY's body into
// Proxy's context, so `y = v` resolves by name against Proxy.y.

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
    // Swapped order: the (Logic*)this cast would put y at Proxy.x's slot.
    uint256 public y;
    uint256 public x;

    function test() public {
        x = 0;
        y = 0;
        address(this).delegatecall(
            abi.encodeWithSignature("setBoth(uint256,uint256)", uint256(10), uint256(20))
        );
        assert(x == 10);
        assert(y == 20);
    }
}
