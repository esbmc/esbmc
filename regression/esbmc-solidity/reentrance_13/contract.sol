//https://github.com/fsainas/contracts-verification-benchmark/blob/main/contracts/bank/
//SPDX-License-Identifier: UNLICENSED
pragma solidity >= 0.8.2;

/// @custom:version conformant to specification
contract Bank {
    mapping (address => uint) balances;

    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }

    function withdraw(uint amount) public {
        /// @custom:preghost function withdraw
        uint old_user_balance = balances[msg.sender];
        require(amount > 0);
        require(amount <= balances[msg.sender]);

        balances[msg.sender] -= amount;
        (bool success,) = msg.sender.call{value: amount}("");
        require(success);
        /// @custom:postghost function withdraw
        uint new_user_balance = balances[msg.sender];
        assert(new_user_balance == old_user_balance - amount);
}
}

interface IBank {
    function deposit() external payable;
    function withdraw(uint amount) external;
}

contract Reproduction {
    IBank public target;
    address public owner;

    constructor(address _target) {
        target = IBank(_target);
        owner = msg.sender;
    }

    function set() external payable {
        require(msg.value > 0);
        target.deposit{value: msg.value}();
    }

    function trigger(uint amount) external {
        require(msg.sender == owner);
        target.withdraw(amount);
    }

    receive() external payable {
        // The guard MUST be able to become false, otherwise this attacker
        // re-enters forever: the innermost withdraw fails its
        // `require(amount <= balances[msg.sender])`, that revert propagates
        // back through every frame's `require(success)` (a low-level call
        // returns ok = !reverted), and the ghost post-condition is never
        // reached.  Same guard as reentrance_2, so the two tests differ only
        // in where Bank does its bookkeeping.
        if (address(target).balance > 0) {
            target.withdraw(msg.value);
        }
    }
}