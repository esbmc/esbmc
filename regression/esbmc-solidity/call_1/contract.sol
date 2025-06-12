// // SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

contract Bank {
    mapping (address => uint) balances;
    address x;
    uint choice;

    function deposit() public payable {
        balances[x] += 1;
    }

    function withdraw(uint amount) public {
        require(amount > 0);
        require(amount <= balances[x]);

        (bool success,) = x.call{value: amount}("");
        balances[x] -= amount;
    }
    function invariant(uint u1, address a) public payable {
        uint currb = balances[a];
        if (choice == 0) {
            deposit();
            ++choice;
        } else {
            withdraw(u1);
        }
        uint newb = balances[a];

        assert(newb == currb);

}
}