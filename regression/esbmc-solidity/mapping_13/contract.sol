// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

contract Bank {
    mapping(address => uint) balances;

    function invariant(uint choice, address a) public payable {
        uint currb = balances[a];
        uint newb = balances[a];
        require(newb != currb);
    }
}

contract T {
    function t() public {
        Bank x = new Bank();
    }
}
