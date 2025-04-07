// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract SimpleContract {
    uint256 public value;

    constructor(uint256 _value) {
        value = _value;
    }
}

contract Factory {
    function getCreationCode() public pure returns (bytes memory) {
        return type(SimpleContract).creationCode; // Returns the creation code
    }

    function getRuntimeCode() public pure returns (bytes memory) {
        return type(SimpleContract).runtimeCode; // Returns the runtime code
    }
}
