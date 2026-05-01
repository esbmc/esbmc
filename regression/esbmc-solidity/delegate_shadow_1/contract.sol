// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// delegatecall storage-context shadow (happy path):
// Logic.setX writes to `x`. Proxy has a `x` field with the same type
// and name. A delegatecall from Proxy to Logic must execute setX against
// Proxy's storage, so Proxy.x must equal 42 afterwards.
//
// The old $delegatecall#0 helper would have run setX against
// _ESBMC_Object_Logic.x (Logic's own static instance), leaving Proxy.x
// untouched. The new delegate-shadow path inlines Logic.setX's body into
// the caller with state-var refs rewritten to Proxy's `x`.
contract Logic {
    uint256 public x;
    function setX(uint256 v) public { x = v; }
}

contract Proxy {
    uint256 public x;

    function test() public {
        x = 0;
        address(this).delegatecall(
            abi.encodeWithSignature("setX(uint256)", uint256(42))
        );
        assert(x == 42);
    }
}
