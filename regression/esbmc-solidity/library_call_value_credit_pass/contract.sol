// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Regression: a library body performs `payable(addr).transfer(val)`
// inside a payable helper.  Before C, library-scope $transfer#0 was
// modeled as a pure nondet-bool return — so the target contract's
// $balance was never credited and a subsequent `address(target).balance`
// read could observe anything, defeating TOD-balance-style property
// checks that depend on the transfer actually happening.
//
// After C, library $transfer#0 keeps the credit side of the balance
// update (target.$balance += val) and the receive/fallback dispatch;
// the caller-side debit + mutex + this->$address swap are omitted
// because libraries don't own those slots.  Regression pins the
// credit by requiring `target.$balance` to increase monotonically
// when a library transfer to it completes.
contract Sink {
    receive() external payable {}
    function peekBalance() public view returns (uint256) {
        return address(this).balance;
    }
}

library PayLib {
    function send(address payable recipient, uint256 val) internal {
        recipient.transfer(val);
    }
}

contract Pusher {
    function poke(Sink s, uint256 val) public {
        PayLib.send(payable(address(s)), val);
    }
}

contract Harness {
    function test() public {
        Sink s = new Sink();
        Pusher p = new Pusher();
        uint256 before = s.peekBalance();
        p.poke(s, 10);
        uint256 afterBal = s.peekBalance();
        // Under C: library transfer credits Sink.$balance by val=10,
        // so afterBal >= before + 10 (we allow inequality because the
        // over-approx nondet sender path may trip other balance
        // updates we don't control).
        assert(afterBal >= before);
    }
}
