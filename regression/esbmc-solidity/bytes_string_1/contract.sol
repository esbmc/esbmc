// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Test bytes(string).length — the pattern used in PropertyTokenisation.sol:
// bytes(properties[propertyID].id).length == 0
contract Test {
    struct Property {
        string id;
        uint256 price;
    }

    mapping(uint256 => Property) public properties;

    constructor() {
        properties[1] = Property("PROP-1", 100);
    }

    function check_exists() public {
        // After constructor, properties[1].id should be non-empty
        assert(bytes(properties[1].id).length > 0);
    }

    function check_not_exists() public {
        // properties[99] was never set, id should be empty
        assert(bytes(properties[99].id).length == 0);
    }
}
