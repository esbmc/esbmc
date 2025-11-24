// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract SimpleMappingStruct {
    // Define a struct
    struct User {
        uint8 age;
    }

    // Mapping from address to User struct
    mapping(address => User) public users;

    // Function to set age
    function setAge(uint8 _age) public {
        //users[msg.sender].age = _age;
    }

    // Function that includes a failing assertion
    function checkAge() public view {
        // This will fail if age is not exactly 100
        assert(users[msg.sender].age == 0);
    }
}
