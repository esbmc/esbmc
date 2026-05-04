// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Adversarial: the argument to setX must be exactly 42 after dispatch.
// This test asserts a wrong expected value (43) and must FAIL, proving
// the dispatch is not silently dropping the argument and leaving x
// unconstrained (a nondet-model would let x be any value and the wrong
// assertion could spuriously succeed for the counter-witness).
contract TypedCallBadValue {
    uint256 public x;
    function setX(uint256 v) public { x = v; }

    function test() public {
        x = 0;
        address(this).call(abi.encodeWithSignature("setX(uint256)", uint256(42)));
        assert(x == 43); // wrong: dispatch writes 42, not 43
    }
}
