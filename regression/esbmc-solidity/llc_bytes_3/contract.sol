// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Low-level call: data.length is a nondet uint256.
// Asserting it equals a specific value (0) should fail,
// as the verifier finds a counterexample where length != 0.
// This test ensures the bytes .length member access produces the
// correct 64-bit type (size_t) and does not cause Z3 sort mismatch.
// VERIFICATION FAILED expected.
contract LLCBytes3 {
    function test() public {
        (bool success, bytes memory data) = address(this).call("");
        assert(data.length == 0);
    }
}
