// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

contract MyContract {
    string public contractName;

    function test() public {
        // type(MyContract).name should return "MyContract"
        contractName = type(MyContract).name;
        // After assignment, contractName should not be empty
        assert(bytes(contractName).length > 0);
    }
}
