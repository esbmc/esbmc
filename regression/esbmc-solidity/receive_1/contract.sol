// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// This contract keeps all Ether sent to it with no way
// to get it back.
contract Sink {
    uint public totalReceived;
    event Received(address, uint);
    receive() external payable {
        totalReceived += msg.value;
        emit Received(msg.sender, msg.value);
    }

    function checkReceived() public view {
        // totalReceived should be non-negative (always true for uint)
        assert(totalReceived >= 0);
    }
}
