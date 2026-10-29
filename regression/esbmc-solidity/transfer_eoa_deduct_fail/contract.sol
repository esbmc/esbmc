// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Negative counterpart: assert balance is INVARIANT across an EOA
// transfer.  After the fix this is provably false (transfer reduces
// balance by `amt`), so verification must fail.
contract Bal {
    constructor() payable {}

    function payAndCheck(address payable to, uint amt) public {
        if (to == address(this)) return; // skip self-transfer
        if (address(this).balance < amt || amt == 0) return;
        uint before = address(this).balance;
        to.transfer(amt);
        uint after_ = address(this).balance;
        // Wrong: balance dropped, so == before is FALSE for amt > 0.
        assert(after_ == before);
    }
}
