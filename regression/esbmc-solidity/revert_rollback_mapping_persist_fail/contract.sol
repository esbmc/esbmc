// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Control for finding #2: a mapping write that is NOT reverted must
// remain visible to the next transaction.  This is a genuine property
// violation (true positive) and must stay FAILED both before and after
// the rollback fix -- it proves the fix only rolls back on revert, not
// on the normal (committed) path.
contract C {
    mapping(uint => uint) m;

    function setNoRevert(uint k) public {
        m[k] = 5;
    }

    function checkBug(uint k) public view {
        assert(m[k] != 5);
    }
}
