// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

contract OwnedToken {
    TokenCreator creator;
    address owner;
    bytes32 name;

    constructor(bytes32 name_) {
        owner = msg.sender;
        creator = TokenCreator(msg.sender);
        name = name_;
    }

    function changeName(bytes32 newName) public {
        if (msg.sender == address(creator))
            name = newName;
    }

    function transfer(address newOwner) public {
        if (msg.sender != owner) return;
        if (creator.isTokenTransferOK(owner, newOwner))
            owner = newOwner;
    }

    function test_name_set() public view {
        assert(name != bytes32(0));
    }
}

contract TokenCreator {
    function changeName(OwnedToken tokenAddress, bytes32 name) public {
        tokenAddress.changeName(name);
    }

    function isTokenTransferOK(address currentOwner, address newOwner)
        public
        pure
        returns (bool ok)
    {
        return currentOwner != newOwner;
    }
}
