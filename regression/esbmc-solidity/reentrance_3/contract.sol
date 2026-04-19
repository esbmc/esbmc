// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

interface ICallWrapper {
    function callwrap(address called) external;
    function modifystorage(uint newdata) external;
}

/// @custom:version reentrant `callwrap`.
contract CallWrapper {
    uint data;

    function callwrap(address called) public {
        /// @custom:preghost function callwrap
        uint _data = data;

        called.call("");

        /// @custom:postghost function callwrap
        assert(_data == data);
    }
    
    function modifystorage(uint newdata) public {
        data = newdata;
    }

}

contract Reproduction {
    CallWrapper public target;
    uint public reentered = 0;

    constructor(address _target) {
        target = CallWrapper(_target);
    }

    function setup(uint newdata) public {
        target.modifystorage(newdata);
    }

    function trigger() public {
        target.callwrap(address(this));
    }

    fallback() external payable {
        if (reentered == 0) {
            reentered = 1;
            target.modifystorage(999);
        }
    }
}


