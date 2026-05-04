// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// encodeWithSelector form: the signature is recovered from the referenced
// function declaration via build_canonical_signature, not from a literal
// string. Must produce the same dispatch behaviour as encodeWithSignature.

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
        assert(x == 42);
    }
}
