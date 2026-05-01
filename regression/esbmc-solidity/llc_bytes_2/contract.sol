// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Low-level call: the success return is modelled as nondet bool.
// The verifier finds a counterexample where success == false,
// so the assertion fails.
// VERIFICATION FAILED expected.
contract LLCBytes2 {
    function test() public {
        (bool success, bytes memory data) = address(this).call("");
        assert(success);
    }
}
