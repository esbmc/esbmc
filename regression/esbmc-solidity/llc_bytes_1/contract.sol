// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Low-level call: bytes return data is modelled as BytesDynamic (nondet).
// data.length is an unsigned nondet value, so data.length >= 0 is always true.
// VERIFICATION SUCCESSFUL expected.
contract LLCBytes1 {
    function test() public {
        (bool success, bytes memory data) = address(this).call("");
        assert(data.length >= 0);
    }
}
