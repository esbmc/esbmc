// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

contract Bank {
    mapping (address => uint) balances;

    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }

    function withdraw(uint amount) public {
        /// @custom:preghost function withdraw
        uint old_user_balance = address(msg.sender).balance;
        require(amount > 0);
        // require(amount <= balances[msg.sender]);

        balances[msg.sender] -= amount;
        (bool success,) = msg.sender.call{value: amount}("");
        require(success);
        /// @custom:postghost function withdraw
        uint new_user_balance = address(msg.sender).balance;
        assert(new_user_balance == old_user_balance + amount);
}
}