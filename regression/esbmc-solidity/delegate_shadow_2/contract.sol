// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Adversarial: the delegatecall must pass through the exact argument
// to the inlined setX body. This test asserts a wrong expected value
// so verification MUST fail. A broken implementation that dropped
// the arg (nondet) could still satisfy the assertion under some model
// by picking x=43 nondeterministically; this ensures we're really
// writing the supplied 42.
contract LogicAdv {
    uint256 public x;
    function setX(uint256 v) public { x = v; }
}

contract ProxyAdv {
    uint256 public x;

    function test() public {
        x = 0;
        address(this).delegatecall(
            abi.encodeWithSignature("setX(uint256)", uint256(42))
        );
        assert(x == 43); // wrong on purpose: the arg is 42, not 43
    }
}
