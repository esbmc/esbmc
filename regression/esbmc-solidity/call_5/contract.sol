// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

contract IBank {
    function deposit() external payable {}
    function withdraw(uint amount) external {}
}

contract Bank {
    mapping(address => uint) balances;

    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }

    function withdraw(uint amount) public {
        /// @custom:preghost function withdraw
        uint old_contract_balance = address(this).balance;
        require(amount > 0);
        require(amount <= balances[msg.sender]);

        (bool success, ) = msg.sender.call{value: amount}("");
        balances[msg.sender] -= amount;
        require(success);
        /// @custom:postghost function withdraw
        uint new_contract_balance = address(this).balance;
        assert(new_contract_balance <= old_contract_balance);
    }
}

contract Reproduction {
    Bank public target;

    constructor(address _target) {
        target = Bank(_target);
        address(target).balance;
    }

    function setup() external payable {
        require(msg.value > 0);
        target.deposit{value: msg.value}();
    }

    function trigger(uint amount) external {
        target.withdraw(amount);
    }

    receive() external payable {
        //target.deposit();
    }
}
