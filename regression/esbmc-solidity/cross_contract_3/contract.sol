// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Test: address-to-contract type cast should preserve the address.
// TokenCreator(msg.sender) stores msg.sender as the singleton's $address.
// Therefore address(creator) == owner should hold in the constructor.

contract OwnedToken {
    TokenCreator creator;
    address owner;

    constructor() {
        owner = msg.sender;
        creator = TokenCreator(msg.sender);
        assert(owner == address(creator));
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
