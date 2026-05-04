// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Signature precision: calling with a non-existent signature must NOT
// dispatch to a differently-named function. If the contract has only
// setX, a call with signature "foo(uint256)" must leave x unchanged.
contract TypedCallMissSig {
    uint256 public x;
    function setX(uint256 v) public { x = v; }

    function test() public {
        x = 7;
        // "foo(uint256)" is not defined; the call must not clobber x.
        address(this).call(abi.encodeWithSignature("foo(uint256)", uint256(99)));
        assert(x == 7);
    }
}
