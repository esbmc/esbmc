// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Regression: a function that combines (a) a modifier, (b) NAMED
// return parameters, and (c) a multi-return tuple, used to crash
// symex with
//   value_set: unknown symbol `sol:@C@<c>@F@<fn>_<mod>@<param>#id`
// Reproduced on SolidiFi buggy_10's `getDetail` (docs/signers tuple,
// validDoc modifier).
//
// Root cause: the modifier expander (`get_func_modifier` in
// src/solidity-frontend/solidity_convert_modifier.cpp) registered the
// named return parameters under the OUTER function's scope only,
// then re-converted the body under the AUX (`fn_mod`) scope.  Body
// references resolved to the aux-scoped identifier which had never
// been moved to the symbol table.
//
// Fix: after switching to the aux scope, re-register each named
// return parameter under the aux function's id and prepend the DECLs
// to the merged body so the inlined body sees them as ordinary local
// variables.
contract C {
    mapping(bytes32 => string) public docs;
    mapping(bytes32 => address[]) public signers;

    modifier validDoc(bytes32 h) {
        require(bytes(docs[h]).length != 0, "x");
        _;
    }

    function getDetail(bytes32 h) public validDoc(h) view
        returns (string memory _doc, address[] memory _signers)
    {
        _doc = docs[h];
        _signers = signers[h];
    }
}

contract H {
    function test(bytes32 h) public {
        C c = new C();
        c.getDetail(h);
    }
}
