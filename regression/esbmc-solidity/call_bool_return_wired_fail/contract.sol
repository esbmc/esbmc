// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// FAIL dual of call_bool_return_wired_pass.  `.call` on an EOA
// (unknown, non-singleton address) correctly routes to the EOA
// fallthrough which returns nondet_bool, so asserting `ok` always
// true is refutable.  Before the tuple-wire-through fix the same
// assertion was ALSO refutable (tuple.success was always nondet
// regardless of target), but the fix now routes tracked-match to
// a definite `true`, leaving only EOA-fallthrough as the nondet
// source.
contract C {
    function test(address payable eoa) public {
        require(eoa != address(this));
        require(eoa != address(0));
        (bool ok, ) = eoa.call{value: 0}("");
        assert(ok);
    }
}
