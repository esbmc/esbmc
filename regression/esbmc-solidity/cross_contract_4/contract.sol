// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Test: inline contract type cast + method call should not crash.
// TokenCreator(address(creator)).isTokenTransferOK(...) exercises
// the inline cast chain unwrapping in get_contract_member_call_expr.
// Cross-contract calls are modeled as nondet, so assert(ok) must FAIL.

contract OwnedToken {
    TokenCreator creator;
    address owner;

    constructor() {
        owner = msg.sender;
        creator = TokenCreator(msg.sender);
    }

    function test_inline_call() public view {
        bool ok = TokenCreator(address(creator)).isTokenTransferOK(owner, owner);
        assert(ok); // must FAIL: cross-contract calls return nondet
    }
}

contract TokenCreator {
    function isTokenTransferOK(address currentOwner, address newOwner)
        public
        pure
        returns (bool ok)
    {
        return currentOwner != newOwner;
    }
}
