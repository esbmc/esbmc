// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

// Regression for "cannot find tuple instance for declaration id N".
//
// A local function-pointer variable of a tuple-returning function type is
// invoked and its result returned. The callee's `referencedDeclaration`
// points at the VariableDeclaration (not a FunctionDefinition), so the
// tuple-return path must fail softly and fall back to nondet-per-member
// rather than log_error.
contract A {
    function _impl(bytes calldata data, address taker)
        internal
        pure
        returns (bool ok, bytes calldata tail)
    {
        ok = (taker != address(0));
        tail = data;
    }

    function probe(bytes calldata data, address taker)
        external
        view
        returns (bool ok, bytes calldata tail)
    {
        function(bytes calldata, address)
            internal
            view
            returns (bool, bytes calldata) fn = _impl;
        return fn(data, taker);
    }
}
