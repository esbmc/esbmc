pragma solidity ^0.8.0;

contract EtherTransferTest {
    address payable public recipient;

    constructor(address payable _recipient) {
        recipient = _recipient;
    }

    // Function that uses transfer
    function transferEther() public payable {
        uint256 halfDay = 0.5 * 10;
        require(msg.value > 10_0.01 ether, "Send more than 0.01 ETH");
        recipient.transfer(0.01 ether);
    }

    // Function that uses send
    function sendEther() public payable {
        require(msg.value > 0.01 ether, "Send more than 0.01 ETH");
        bool success = recipient.send(0.01 ether);
        require(success, "Send failed");
    }
}

