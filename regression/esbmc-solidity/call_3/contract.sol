// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

contract CallWrapper {
    uint data;

    function callwrap(address called) public {
        /// @custom:preghost function callwrap
        uint _balance = address(this).balance;

        called.call("");

        /// @custom:postghost function callwrap
        assert(_balance == address(this).balance);
    }
    
    function modifystorage(uint newdata) public {
        data = newdata;
    }

}