// GPT4o-Generate
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract Auction {
    address payable public seller;
    uint public auctionEndTime;
    string public item;

    address public highestBidder;
    uint public highestBid;

    mapping(address => uint) public pendingReturns;

    bool public ended;

    event HighestBidIncreased(address bidder, uint amount);
    event AuctionEnded(address winner, uint amount);

    constructor(uint _biddingTime, string memory _item) {
        seller = payable(msg.sender);
        auctionEndTime = block.timestamp + _biddingTime;
        item = _item;
    }

    function bid() external payable {
        require(block.timestamp <= auctionEndTime, "Auction already ended.");
        require(msg.value > highestBid, "There already is a higher bid.");

        if (highestBid != 0) {
            pendingReturns[highestBidder] += highestBid;
        }

        highestBidder = msg.sender;
        highestBid = msg.value;
        emit HighestBidIncreased(msg.sender, msg.value);
    }

    function withdraw() external returns (bool) {
        uint amount = pendingReturns[msg.sender];
        require(amount > 0, "No funds to withdraw.");

        pendingReturns[msg.sender] = 0;

        (bool success, ) = payable(msg.sender).call{value: amount}("");
        if (!success) {
            pendingReturns[msg.sender] = amount;
            return false;
        }
        return true;
    }

    function auctionEnd() external {
        require(block.timestamp >= auctionEndTime, "Auction not yet ended.");
        require(!ended, "auctionEnd has already been called.");

        ended = true;
        emit AuctionEnded(highestBidder, highestBid);

        (bool success, ) = seller.call{value: highestBid}("");
        require(success, "Transfer to seller failed.");
    }
}
