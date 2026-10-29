// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// `.send` to a tracked target models failure as `ok = !reverted` (same as
// `.call`).  A reverting receive makes `send` return false, so `assert(ok)`
// is refutable.  Dual: send_callee_revert_pass.
contract C {
    bool reject;
    constructor(bool r) { reject = r; }
    receive() external payable { require(!reject, "no"); }
    function test() public {
        require(address(this).balance >= 1);
        bool ok = payable(address(this)).send(1);
        assert(ok);
    }
}
