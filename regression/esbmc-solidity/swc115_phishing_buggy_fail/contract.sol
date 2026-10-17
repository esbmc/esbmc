// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// SWC-115: Authorization through tx.origin.
// The `setSecret` guard is `tx.origin == owner`, which is satisfied
// not only when the owner calls directly but also when the owner is
// the original EOA on a relay chain `owner -> attacker -> Vault`,
// in which case `msg.sender` is the attacker contract.  The oracle
// `assert(msg.sender == owner)` therefore must FAIL to surface the
// phishing path that the wrong guard authorises.
contract Vault {
    address public owner;
    uint256 public secret;

    constructor() {
        owner = msg.sender;
    }

    function setSecret(uint256 v) external {
        require(tx.origin == owner);
        assert(msg.sender == owner);
        secret = v;
    }
}
