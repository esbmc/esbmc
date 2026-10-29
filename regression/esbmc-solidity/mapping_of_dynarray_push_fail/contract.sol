// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Fail-side of `mapping_of_dynarray_push_pass`: C exposes the same
// `mapping(K => T[]).push` shape that the frontend used to crash on
// during conversion.  Contract T instantiates C and then fails an
// assertion — the frontend must convert both contracts without the
// substr out_of_range crash, and then symex must deliver the failing
// assertion to the solver.
//
// NB: we don't call pushOne here because the end-to-end
// mapping-of-dynarray semantics (write-through of the relocated
// heap pointer back into the mapping storage) is a separate,
// tracked limitation — exercising it tends to collapse the VCC set
// through it.  This test pairs with the pass case specifically on
// the conversion-side crash.

contract C {
    mapping(address => uint256[]) m;
    function pushOne(address a, uint256 x) public { m[a].push(x); }
}

contract T {
    function test() public {
        C c = new C();
        c;  // prevent slicing out the allocation
        assert(false);
    }
}
