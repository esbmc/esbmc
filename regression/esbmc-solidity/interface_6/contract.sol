// SPDX-License-Identifier: MIT
pragma solidity >=0.5.0;

interface IUserRegistry {
    function register(address user) external;
    function isRegistered(address user) external view returns (bool);
}

contract UserRegistry is IUserRegistry {
    

    function register(address user) external  {
        _registeredUsers[user] = true;
    }

    function isRegistered(address user) external view  returns (bool) {
        return _registeredUsers[user];
    }
    mapping(address => bool) private _registeredUsers;
}
