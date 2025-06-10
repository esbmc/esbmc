// // SPDX-License-Identifier: GPL-3.0
// pragma solidity >=0.8.0;

// contract Bank {
//     mapping (address => uint) balances;

//     function deposit() public payable {
//         balances[msg.sender] += msg.value;
//     }

//     function withdraw(uint amount) public {
//         require(amount > 0);
//         require(amount <= balances[msg.sender]);

//         (bool success,) = msg.sender.call{value: amount}("");
//         balances[msg.sender] -= amount;
//         require(success);
//     }
//     function invariant(uint choice, uint u1, address a) public payable {
//         uint currb = balances[a];
//         if (choice == 0) {
//             deposit();
//         } else if (choice == 1) {
//             withdraw(u1);
//         } else {
//             require(false);
//         }
//         uint newb = balances[a];

//         require(newb < currb);
//         assert(choice == 1);
//         assert(msg.sender == a);
// }
// }

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